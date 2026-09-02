/* argmax_with_index_fp64 -- TSVC tsvc_2_5 (s315): running maximum carrying BOTH
 * the value and its index (first/leftmost maximum, strict >).
 *
 * Single pass over `a`. Per thread: 8 independent scalar max-streams carried in
 * 8 AVX-512 lanes (element j+8g updates lane j), so the whole per-thread sweep
 * is a branchless vector loop bound by memory bandwidth. Threads get contiguous
 * spans; partials combine with (max value, min index on tie), which is
 * associative and reproduces the leftmost-argmax exactly.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#define PART_CAP 512

struct part { double v; int64_t i; char pad[48]; }; /* 64B slot: no false sharing */
static _Alignas(64) struct part part_[PART_CAP];

#if defined(__AVX512F__)
/* Argmax over [off, end) of p, where (end-off) is a positive multiple of 8.
 * Returns (best value, first index at which it occurs). */
static void argmax_8lane(const double * __restrict p, int64_t off, int64_t end,
                         double *bv, int64_t *bi)
{
    const __m512i lane07 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    const __m512i eight = _mm512_setr_epi64(8, 8, 8, 8, 8, 8, 8, 8);
    const __m512i big = _mm512_setr_epi64(
        0x7fffffffffffffffLL, 0x7fffffffffffffffLL, 0x7fffffffffffffffLL,
        0x7fffffffffffffffLL, 0x7fffffffffffffffLL, 0x7fffffffffffffffLL,
        0x7fffffffffffffffLL, 0x7fffffffffffffffLL);

    __m512d mv = _mm512_loadu_pd(p + off); /* lane j = p[off+j] */
    int64_t ob[8] = { off, off, off, off, off, off, off, off };
    __m512i idx = _mm512_add_epi64(lane07, _mm512_loadu_si512((const void *)ob));
    __m512i rvec = idx; /* first index per lane (absolute), set on first beat  */
    const double *qe = p + end - 8;
    for (const double *q = p + off + 8; q <= qe; q += 8) {
        __m512d v = _mm512_loadu_pd(q);
        __mmask8 beat = _mm512_cmp_pd_mask(v, mv, _CMP_GT_OQ);
        mv = _mm512_max_pd(v, mv);
        idx = _mm512_add_epi64(idx, eight); /* idx now = this group's */
        rvec = _mm512_mask_blend_epi64(beat, rvec, idx);
    }
    /* reduce across lanes: max value, then first index among the max lanes */
    double M = _mm512_reduce_max_pd(mv);
    __mmask8 eq = _mm512_cmpeq_pd_mask(mv, _mm512_set1_pd(M));
    /* mask_blend selects the THIRD arg where the bit is set */
    __m512i cand = _mm512_mask_blend_epi64(eq, big, rvec);
    *bv = M;
    *bi = (int64_t)_mm512_reduce_min_epu64(cand);
}
#else
static void argmax_8lane(const double * __restrict p, int64_t off, int64_t end,
                         double *bv, int64_t *bi)
{
    double v = p[off];
    int64_t i = off;
    for (int64_t k = off + 1; k < end; k++)
        if (p[k] > v) { v = p[k]; i = k; }
    *bv = v;
    *bi = i;
}
#endif

static void chunk_argmax(const double * __restrict a, int64_t base, int64_t size,
                         double *bv, int64_t *bi)
{
    const double *p = a + base;
    double v = p[0];
    int64_t i = base;
    int64_t k = 1;
    /* align up to 64B (<= 7 elements) so the vector region is aligned */
    int64_t off = (int64_t)(((64 - ((uintptr_t)p & 63)) & 63) >> 3);
    if (off > size) off = size;
    for (; k < off; k++)
        if (p[k] > v) { v = p[k]; i = base + k; }
    int64_t vstart = base + off;
    int64_t vend = vstart + (((base + (size & ~7)) - vstart) & ~7); /* len%8==0 */
    if (vend - vstart >= 8) {
        double mv; int64_t mi;
        argmax_8lane(a, vstart, vend, &mv, &mi);
        if (mv > v) { v = mv; i = mi; } /* prefix precedes: keep earlier on tie */
    }
    {
        int64_t k0 = vend - base;
        if (k0 < off) k0 = off; /* never re-enter the peeled prefix */
        for (k = k0; k < size; k++)
            if (p[k] > v) { v = p[k]; i = base + k; }
    }
    *bv = v;
    *bi = i;
}

void argmax_with_index_fp64(const double *a, int64_t *out_index, double *out_value,
                            int64_t LEN_1D, unsigned char *workspace,
                            int64_t workspace_size)
{
    (void)workspace; (void)workspace_size;
    int64_t n = LEN_1D;
    if (n <= 1) {
        out_value[0] = (n > 0) ? a[0] : 0.0;
        out_index[0] = 0;
        return;
    }

    int maxt = omp_get_max_threads();
    int nt = (maxt < PART_CAP) ? maxt : PART_CAP;

    /* below this size the fork/join costs more than the work */
    if (nt <= 1 || n < 65536) {
        double v = a[0];
        int64_t i = 0;
        for (int64_t k = 1; k < n; k++)
            if (a[k] > v) { v = a[k]; i = k; }
        out_value[0] = v;
        out_index[0] = i;
        return;
    }

    int64_t chunk = n / nt;
    int64_t rem = n % nt;
    #pragma omp parallel num_threads(nt)
    {
        int t = omp_get_thread_num();
        int64_t base = t * chunk + ((int64_t)t < rem ? (int64_t)t : rem);
        int64_t size = chunk + ((int64_t)t < rem ? 1 : 0);
        double pv; int64_t pi;
        chunk_argmax(a, base, size, &pv, &pi);
        part_[t].v = pv;
        part_[t].i = pi;
    }
    double bv = part_[0].v;
    int64_t bi = part_[0].i;
    for (int t = 1; t < nt; t++) {
        if (part_[t].v > bv) { bv = part_[t].v; bi = part_[t].i; }
        else if (part_[t].v == bv && part_[t].i < bi) bi = part_[t].i;
    }
    out_value[0] = bv;
    out_index[0] = bi;
}
