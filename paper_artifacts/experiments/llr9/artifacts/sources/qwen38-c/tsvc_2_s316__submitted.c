/* TSVC s316: result[0] = min(a[0..LEN_1D)) with left-to-right strict '<' semantics.
 * OpenMP: each thread streams a contiguous slab with 4 independent 256-bit
 * running-min chains; combined via reduction(min:).
 * NaN safety: vminpd(x, v) ignores a NaN in v (2nd operand); a[0]==NaN
 * special-cased (reference keeps it forever).
 */
#include <stdint.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>

static inline double vmin_scalar_vec(__m256d a0, __m256d a1, __m256d a2, __m256d a3)
{
    __m128d m = _mm_min_pd(_mm256_castpd256_pd128(a0), _mm256_extractf128_pd(a0, 1));
    m = _mm_min_pd(m, _mm_min_pd(_mm256_castpd256_pd128(a1), _mm256_extractf128_pd(a1, 1)));
    m = _mm_min_pd(m, _mm_min_pd(_mm256_castpd256_pd128(a2), _mm256_extractf128_pd(a2, 1)));
    m = _mm_min_pd(m, _mm_min_pd(_mm256_castpd256_pd128(a3), _mm256_extractf128_pd(a3, 1)));
    m = _mm_min_pd(m, _mm_unpackhi_pd(m, m));
    return _mm_cvtsd_f64(m);
}

static inline double slab_min(const double *restrict p, int64_t len, double seed)
{
    __m256d x0 = _mm256_set1_pd(seed);
    __m256d x1 = x0, x2 = x0, x3 = x0;
    int64_t i = 0, lim = len - 15;
    for (; i < lim; i += 16) {
        __m256d v0 = _mm256_loadu_pd(p + i);
        __m256d v1 = _mm256_loadu_pd(p + i + 4);
        __m256d v2 = _mm256_loadu_pd(p + i + 8);
        __m256d v3 = _mm256_loadu_pd(p + i + 12);
        x0 = _mm256_min_pd(x0, v0);
        x1 = _mm256_min_pd(x1, v1);
        x2 = _mm256_min_pd(x2, v2);
        x3 = _mm256_min_pd(x3, v3);
    }
    double r = vmin_scalar_vec(x0, x1, x2, x3);
    for (; i < len; i++)
        if (p[i] < r) r = p[i];
    return r;
}

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result,
                      int64_t LEN_1D, uint8_t *workspace, int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    int64_t n = LEN_1D;
    if (n <= 0) { *result = 0.0; return; }
    double first = a[0];
    if (n == 1 || isnan(first)) { *result = first; return; }

    double best = first;
    if (n >= (1 << 17)) {
        #pragma omp parallel reduction(min:best)
        {
            int64_t nt = omp_get_num_threads();
            int64_t t  = omp_get_thread_num();
            int64_t per = (n + nt - 1) / nt;
            int64_t i0 = 1 + t * per;
            int64_t i1 = i0 + per;
            if (i1 > n) i1 = n;
            if (i0 < i1) best = slab_min(a + i0, i1 - i0, first);
        }
    } else {
        best = slab_min(a + 1, n - 1, first);
    }
    *result = best;
}
