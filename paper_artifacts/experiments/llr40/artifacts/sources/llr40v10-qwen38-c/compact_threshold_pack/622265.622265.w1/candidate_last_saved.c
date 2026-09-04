/* Stream compaction: pack src[i]*weight[i] for every src[i] > 0, in source order.
 *
 * Two blocked passes:
 *   1) per-block survivor counts (vectorized compare + popcount),
 *   2) exclusive scan of the (few) block counts, then per-block compaction
 *      writing to packed at the scanned offsets (vectorized masked multiply +
 *      AVX-512 compress store on supporting CPUs).
 * Ordering is preserved: blocks are packed in order, each block sequentially.
 */
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512ER__)
#include <immintrin.h>
#define CTP_AVX512 1
#else
#define CTP_AVX512 0
#endif

static inline int64_t ctp_count(const double *s, int64_t n) {
#if CTP_AVX512
    int64_t c = 0, i = 0;
    const __m512d zero = _mm512_setzero_pd();
    for (; i + 8 <= n; i += 8)
        c += (int64_t)_mm_popcnt_u8(_mm512_cmp_pd_mask(_mm512_loadu_pd(s + i), zero,
                                                       _CMP_GT_OQ));
    for (; i < n; i++) c += (s[i] > 0.0);
    return c;
#else
    int64_t c = 0;
    for (int64_t i = 0; i < n; i++) c += (s[i] > 0.0);
    return c;
#endif
}

static inline void ctp_pack(const double *s, const double *w, double *p, int64_t n) {
#if CTP_AVX512
    const __m512d zero = _mm512_setzero_pd();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d vs = _mm512_loadu_pd(s + i);
        __mmask8 m = _mm512_cmp_pd_mask(vs, zero, _CMP_GT_OQ);
        if (m) {
            __m512d vw = _mm512_loadu_pd(w + i);
            _mm512_mask_compressstoreu_pd(p, m, _mm512_mul_pd(vs, vw));
            p += _mm_popcnt_u8(m);
        }
    }
    for (; i < n; i++) {
        if (s[i] > 0.0) *p++ = s[i] * w[i];
    }
#else
    for (int64_t i = 0; i < n; i++) {
        if (s[i] > 0.0) *p++ = s[i] * w[i];
    }
#endif
}

static void ctp_impl(const double *restrict src, const double *restrict weight,
                     double *restrict packed, int64_t *restrict out_count,
                     const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        *out_count = 0;
        return;
    }

    const int nt = (int)omp_get_max_threads();

    /* Serial-vector path for small inputs: no team spawn, no malloc. */
    if (LEN_1D < (1 << 17) || nt <= 1) {
        ctp_pack(src, weight, packed, LEN_1D);
        *out_count = ctp_count(src, LEN_1D);
        return;
    }

    /* Choose block count: ~8 blocks/thread, but keep blocks >= 8192 elems. */
    int64_t nb = (int64_t)nt * 8;
    int64_t nb_max = LEN_1D >> 13; /* 8192 per block */
    if (nb_max < 1) nb_max = 1;
    if (nb > nb_max) nb = nb_max;

    int64_t bs = (LEN_1D + nb - 1) / nb;
    bs = (bs + 63) & ~63LL;
    nb = (LEN_1D + bs - 1) / bs;

    static int64_t *counts = NULL;
    static int64_t counts_cap = 0;
    if (nb > counts_cap) {
        int64_t nc = nb > 256 ? nb : 256;
        int64_t *tmp = (int64_t *)realloc(counts, nc * sizeof(int64_t));
        if (tmp) {
            counts = tmp;
            counts_cap = nc;
        }
    }
    if (!counts) { /* fall back to a fully serial pass */
        ctp_pack(src, weight, packed, LEN_1D);
        *out_count = ctp_count(src, LEN_1D);
        return;
    }

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int64_t b = 0; b < nb; b++) {
            int64_t beg = b * bs;
            int64_t end = beg + bs;
            if (end > LEN_1D) end = LEN_1D;
            counts[b] = ctp_count(src + beg, end - beg);
        }
        /* barrier at end of the omp for above; thread 0 does the tiny
         * exclusive scan; single's own end barrier publishes it */
        #pragma omp single
        {
            int64_t acc = 0;
            for (int64_t b = 0; b < nb; b++) {
                int64_t c = counts[b];
                counts[b] = acc;
                acc += c;
            }
            out_count[0] = acc;
        }
        #pragma omp for schedule(static)
        for (int64_t b = 0; b < nb; b++) {
            int64_t beg = b * bs;
            int64_t end = beg + bs;
            if (end > LEN_1D) end = LEN_1D;
            ctp_pack(src + beg, weight + beg, packed + counts[b], end - beg);
        }
    }
}

void compact_threshold_pack_fp64(const double *restrict src, const double *restrict weight,
                                 double *restrict packed, int64_t *restrict out_count,
                                 const int64_t LEN_1D) {
    ctp_impl(src, weight, packed, out_count, LEN_1D);
}

void compact_threshold_pack(const double *restrict src, const double *restrict weight,
                            double *restrict packed, int64_t *restrict out_count,
                            const int64_t LEN_1D) {
    ctp_impl(src, weight, packed, out_count, LEN_1D);
}
