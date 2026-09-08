#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

/* Stream compaction: packed[n++] = src[i]*weight[i] for src[i] > 0.
 *
 * Fused per-block pipeline: each thread counts a block (single src stream
 * from DRAM, lines land in its own L3), then packs that same block once the
 * global base offset is published by the serial claim chain -- so the pack
 * re-read of src hits local L3 and DRAM sees src only once. Weight is
 * loaded only for live groups; packed is written once with NT stores.
 */

#ifndef BLOCK_ELEMS
#define BLOCK_ELEMS 32768LL           /* 256 KB of src */
#endif
#define SMALL_LIMIT (1LL << 15)       /* below: serial single pass */

#ifdef __AVX512F__

static inline int64_t grp_0ff(__m512d pr, double *p, int nt, int64_t off)
{
    if (nt && (off & 7) == 0)
        _mm512_stream_pd(p + off, pr);
    else
        _mm512_storeu_pd(p + off, pr);
    return off + 8;
}
static inline int64_t grp_mix(__m512d pr, int m, double *p, int64_t off)
{
    double buf[8];
    _mm512_storeu_pd(buf, pr);
    if (m & 1) p[off++] = buf[0];
    if (m & 2) p[off++] = buf[1];
    if (m & 4) p[off++] = buf[2];
    if (m & 8) p[off++] = buf[3];
    if (m & 16) p[off++] = buf[4];
    if (m & 32) p[off++] = buf[5];
    if (m & 64) p[off++] = buf[6];
    if (m & 128) p[off++] = buf[7];
    return off;
}

/* count survivors in [s, s+len) */
static int64_t scan_count(const double *s, int64_t len)
{
    int64_t c = 0;
    int64_t i = 0;
    const int64_t nvec = len & ~7LL;
    const __m512d zero = _mm512_setzero_pd();
    for (; i < nvec; i += 32) {
        if (i + 32 < nvec) {
            __builtin_prefetch(s + i + 128, 0, 1);
            __builtin_prefetch(s + i + 256, 0, 1);
        }
        __m512d v0 = _mm512_loadu_pd(s + i);
        __m512d v1 = _mm512_loadu_pd(s + i + 8);
        __m512d v2 = _mm512_loadu_pd(s + i + 16);
        __m512d v3 = _mm512_loadu_pd(s + i + 24);
        c += (int64_t)__builtin_popcountll((unsigned long long)_mm512_cmp_pd_mask(v0, zero, _CMP_GT_OQ));
        c += (int64_t)__builtin_popcountll((unsigned long long)_mm512_cmp_pd_mask(v1, zero, _CMP_GT_OQ));
        c += (int64_t)__builtin_popcountll((unsigned long long)_mm512_cmp_pd_mask(v2, zero, _CMP_GT_OQ));
        c += (int64_t)__builtin_popcountll((unsigned long long)_mm512_cmp_pd_mask(v3, zero, _CMP_GT_OQ));
    }
    for (; i < nvec; i += 8)
        c += (int64_t)__builtin_popcountll((unsigned long long)_mm512_cmp_pd_mask(_mm512_loadu_pd(s + i), zero, _CMP_GT_OQ));
    for (; i < len; i++) c += (s[i] > 0.0);
    return c;
}

/* dense pack of one block; weight loaded only for live groups */
static void pack_block(const double *s, const double *w, double *p,
                       int64_t len, int64_t off)
{
    int64_t i = 0;
    const int64_t nvec = len & ~7LL;
    const __m512d zero = _mm512_setzero_pd();
    const int nt_store = ((uintptr_t)p & 63) == 0;
    for (; i + 16 <= nvec; i += 16) {
        __m512d v0 = _mm512_loadu_pd(s + i);
        __m512d v1 = _mm512_loadu_pd(s + i + 8);
        const int m0 = (int)_mm512_cmp_pd_mask(v0, zero, _CMP_GT_OQ);
        const int m1 = (int)_mm512_cmp_pd_mask(v1, zero, _CMP_GT_OQ);
        if (m0 == 0xff)
            off = grp_0ff(_mm512_mul_pd(v0, _mm512_loadu_pd(w + i)), p, nt_store, off);
        else if (m0)
            off = grp_mix(_mm512_mul_pd(v0, _mm512_loadu_pd(w + i)), m0, p, off);
        if (m1 == 0xff)
            off = grp_0ff(_mm512_mul_pd(v1, _mm512_loadu_pd(w + i + 8)), p, nt_store, off);
        else if (m1)
            off = grp_mix(_mm512_mul_pd(v1, _mm512_loadu_pd(w + i + 8)), m1, p, off);
    }
    for (; i < nvec; i += 8) {
        __m512d v = _mm512_loadu_pd(s + i);
        const int m = (int)_mm512_cmp_pd_mask(v, zero, _CMP_GT_OQ);
        if (m == 0) continue;
        if (m == 0xff)
            off = grp_0ff(_mm512_mul_pd(v, _mm512_loadu_pd(w + i)), p, nt_store, off);
        else
            off = grp_mix(_mm512_mul_pd(v, _mm512_loadu_pd(w + i)), m, p, off);
    }
    for (; i < len; i++)
        if (s[i] > 0.0) p[off++] = s[i] * w[i];
    if (nt_store) _mm_sfence();
}

static void pack_serial(const double *s, const double *w, double *p,
                        int64_t N, int64_t *out_count)
{
    int64_t off = 0;
    int64_t i = 0;
    const int64_t nvec = N & ~7LL;
    const __m512d zero = _mm512_setzero_pd();
    for (; i < nvec; i += 8) {
        __m512d v = _mm512_loadu_pd(s + i);
        const int m = (int)_mm512_cmp_pd_mask(v, zero, _CMP_GT_OQ);
        if (m == 0) continue;
        if (m == 0xff)
            off = grp_0ff(_mm512_mul_pd(v, _mm512_loadu_pd(w + i)), p, 1, off);
        else
            off = grp_mix(_mm512_mul_pd(v, _mm512_loadu_pd(w + i)), m, p, off);
    }
    for (; i < N; i++)
        if (s[i] > 0.0) p[off++] = s[i] * w[i];
    *out_count = off;
}

/* extend the published prefix as far as block counts allow */
static void try_advance(int64_t nblocks, const int64_t *bcnt, int64_t *base,
                        int64_t *ready)
{
    for (;;) {
        const int64_t r = __atomic_load_n(ready, __ATOMIC_ACQUIRE);
        if (r + 1 >= nblocks) return;
        if (__atomic_load_n(&bcnt[r + 1], __ATOMIC_ACQUIRE) < 0) return;
        /* every thread seeing the same r computes the same value, so the
         * pre-CAS store is safe; it must precede the CAS that publishes r+1 */
        const int64_t val = (r >= 0) ? base[r] + bcnt[r] : 0;
        base[r + 1] = val;
        int64_t expected = r;
        __atomic_compare_exchange_n(ready, &expected, r + 1, 0,
                                    __ATOMIC_RELEASE, __ATOMIC_RELAXED);
        /* loop either way: on success keep extending the chain */
    }
}

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 const double *restrict src,
                                 const double *restrict weight,
                                 const int64_t LEN_1D)
{
    const int64_t N = LEN_1D;
    if (N <= 0) { *out_count = 0; return; }

    if (N < SMALL_LIMIT) {
        pack_serial(src, weight, packed, N, out_count);
        return;
    }

    int nt = omp_get_max_threads();
    if (nt < 1) nt = 1;
    if (nt > 32) nt = 32;

    const int64_t nblocks = (N + BLOCK_ELEMS - 1) / BLOCK_ELEMS;
    if (nblocks <= 4 && nt > nblocks) nt = (int)nblocks;

    int64_t *bcnt = (int64_t *)malloc(nblocks * sizeof(int64_t));
    int64_t *base = (int64_t *)malloc(nblocks * sizeof(int64_t));
    memset(bcnt, 0xFF, nblocks * sizeof(int64_t));   /* -1 = not counted */
    static int64_t ready = -1;
    ready = -1;
    int64_t next_cnt = 0;

    #pragma omp parallel num_threads(nt)
    {
        for (;;) {
            const int64_t k = __atomic_fetch_add(&next_cnt, 1, __ATOMIC_RELAXED);
            if (k >= nblocks) break;
            int64_t s0 = k * BLOCK_ELEMS;
            int64_t len = N - s0;
            if (len > BLOCK_ELEMS) len = BLOCK_ELEMS;
            /* count: src stream fills our own L3 */
            const int64_t c = scan_count(src + s0, len);
            __atomic_store_n(&bcnt[k], c, __ATOMIC_RELEASE);
            try_advance(nblocks, bcnt, base, &ready);
            /* wait for the global base of this block, then pack it while
             * its src is still hot in local L3 */
            while (k > __atomic_load_n(&ready, __ATOMIC_ACQUIRE))
                _mm_pause();
            pack_block(src + s0, weight + s0, packed, len, base[k]);
        }
    }

    *out_count = base[nblocks - 1] + bcnt[nblocks - 1];
    free(bcnt);
    free(base);
}

#else
void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 const double *restrict src,
                                 const double *restrict weight,
                                 const int64_t LEN_1D)
{
    const int64_t N = LEN_1D;
    int64_t n = 0;
    for (int64_t i = 0; i < N; i++) n += (src[i] > 0.0);
    int64_t off = 0;
    for (int64_t i = 0; i < N; i++)
        if (src[i] > 0.0) packed[off++] = src[i] * weight[i];
    *out_count = n;
}
#endif
