#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* TSVC tsvc_2 s3110: bb[0] = max(aa) + first_x + first_y where (first_x, first_y)
 * is the FIRST occurrence (row-major) of the global maximum.
 * Single-pass: AVX-512 chunked argmax per thread + serial combine. */

static inline double cmax8(__m512d a)
{
    __m256d lo = _mm512_castpd512_pd256(a);
    __m256d hi = _mm512_extractf64x4_pd(a, 1);
    __m256d m = _mm256_max_pd(lo, hi);
    __m128d l2 = _mm256_extractf128_pd(m, 0);
    __m128d h2 = _mm256_extractf128_pd(m, 1);
    __m128d n = _mm_max_pd(l2, h2);
    __m128d v = _mm_max_pd(n, _mm_unpackhi_pd(n, n));
    return _mm_cvtsd_f64(v);
}

void tsvc_2_s3110_fp64(double *aa, double *bb, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace; (void)workspace_bytes;
    if (LEN_2D <= 0) { bb[0] = 0.0; return; }
    int64_t total = LEN_2D * LEN_2D;

    int nt = omp_get_max_threads();
    if (nt < 1) nt = 1;
    if (nt > 128) nt = 128;

    static double part_max[128];
    static int64_t part_idx[128];

    int64_t chunk = (total + nt - 1) / nt;
    int k = (int)((total + chunk - 1) / chunk);

    #pragma omp parallel for schedule(static, 1)
    for (int t = 0; t < nt; t++) {
        if (t >= k) continue;
        int64_t start = (int64_t)t * chunk;
        int64_t end = start + chunk;
        if (end > total) end = total;
        const double *p = aa + start;
        int64_t n = end - start;

        if (n < 8) {
            double best = p[0];
            int64_t idx = 0;
            for (int64_t i = 1; i < n; i++)
                if (p[i] > best) { best = p[i]; idx = i; }
            part_max[t] = best;
            part_idx[t] = start + idx;
            continue;
        }

        double best = p[0];
        int64_t idx = 0;
        int64_t i = 1;
        for (; i + 8 <= n; i += 8) {
            __m512d a = _mm512_loadu_pd(p + i);
            int m = _mm512_cmp_pd_mask(a, _mm512_set1_pd(best), _CMP_GT_OQ);
            if (m) {
                double d = cmax8(a);
                int e = _mm512_cmp_pd_mask(a, _mm512_set1_pd(d), _CMP_GE_OQ);
                idx = i + __builtin_ctz(e);
                best = d;
            }
        }
        for (; i < n; i++)
            if (p[i] > best) { best = p[i]; idx = i; }
        part_max[t] = best;
        part_idx[t] = start + idx;
    }

    double g = part_max[0];
    int64_t gi = part_idx[0];
    for (int t = 1; t < k; t++) {
        if (part_max[t] > g) { g = part_max[t]; gi = part_idx[t]; }
        else if (part_max[t] == g && part_idx[t] < gi) gi = part_idx[t];
    }
    int64_t xi = gi / LEN_2D, yi = gi % LEN_2D;
    bb[0] = g + (double)xi + (double)yi;
}
