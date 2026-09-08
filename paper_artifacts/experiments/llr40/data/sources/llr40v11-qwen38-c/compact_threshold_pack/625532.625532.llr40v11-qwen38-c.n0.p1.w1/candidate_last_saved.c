/* Stream compaction: packed[0..n) = src[i]*weight[i] for src[i] > 0 (order preserved).
 * Deterministic two-pass: per-chunk counts, exclusive scan over chunk counts, ordered
 * scatter. Pass 2 is 16-wide AVX-512 with mask fast paths to avoid per-element branches. */
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 const double *restrict src,
                                 const double *restrict weight,
                                 const int64_t LEN_1D,
                                 uint8_t *restrict workspace,
                                 const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    const int64_t n = LEN_1D;
    if (n <= 0) {
        out_count[0] = 0;
        return;
    }

    int nt = (int)omp_get_max_threads();
    if (nt < 1) nt = 1;
    const int64_t chunk = (((n + (int64_t)4 * nt - 1) / ((int64_t)4 * nt)) + 63) & ~63LL;
    const int64_t nchunks = (n + chunk - 1) / chunk;

    int64_t *counts = (int64_t *)malloc((size_t)(nchunks + 1) * sizeof(int64_t));

#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nchunks; ++c) {
        const int64_t i0 = c * chunk;
        int64_t i1 = i0 + chunk;
        if (i1 > n) i1 = n;
        int64_t s = 0;
        for (int64_t i = i0; i < i1; ++i) s += (src[i] > 0.0);
        counts[c] = s;
    }

    int64_t total = 0;
    for (int64_t c = 0; c < nchunks; ++c) {
        int64_t t = counts[c];
        counts[c] = total;
        total += t;
    }
    out_count[0] = total;

    const __m512d zero = _mm512_setzero_pd();

#pragma omp parallel for schedule(static)
    for (int64_t c = 0; c < nchunks; ++c) {
        int64_t s = counts[c];
        const int64_t i0 = c * chunk;
        int64_t i1 = i0 + chunk;
        if (i1 > n) i1 = n;
        int64_t i = i0;
        const int64_t i16 = i1 - 15;
        for (; i < i16; i += 16) {
            const __m512d v = _mm512_loadu_pd(src + i);
            const __mmask16 k = _mm512_cmp_pd_mask(v, zero, _CMP_GT_OQ);
            if (k == 0) continue;
            const __m512d p = _mm512_mul_pd(v, _mm512_loadu_pd(weight + i));
            if (k == 0xFFFF) {
                _mm512_storeu_pd(packed + s, p);
                s += 16;
                continue;
            }
            double tmp[16];
            _mm512_storeu_pd(tmp, p);
            int m = (int)k;
            while (m) {
                int b = __builtin_ctz((unsigned)m);
                packed[s++] = tmp[b];
                m &= m - 1;
            }
        }
        for (; i < i1; ++i) {
            double v = src[i];
            if (v > 0.0) packed[s++] = v * weight[i];
        }
    }

    free(counts);
}
