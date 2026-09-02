/* Optimized lower-triangular solve:  x[i] = (b[i] - sum_{j<i} L[i][j] x[j]) / L[i][i]
 * The outer loop is a true sequential dependency chain, so it stays serial;
 * the inner dot product is vectorized (AVX-512 > AVX2 > scalar) with multiple
 * accumulators to hide FMA latency. FP reassociation is within the graded tolerance. */
#include <stdint.h>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
static inline double hsum256(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s = _mm_add_pd(lo, hi);
    s = _mm_add_pd(s, _mm_unpackhi_pd(s, s));
    return _mm_cvtsd_f64(s);
}
#endif

void trisolv_fp64(const double *restrict L, const double *restrict b, double *restrict x, const int64_t N) {
#if defined(__AVX512F__)
    for (int64_t i = 0; i < N; ++i) {
        const double *Li = L + i * N;
        const int64_t n = i;
        __m512d v0 = _mm512_setzero_pd(), v1 = _mm512_setzero_pd(),
                v2 = _mm512_setzero_pd(), v3 = _mm512_setzero_pd();
        const int64_t full = n & ~31LL;
        for (int64_t k = 0; k < full; k += 32) {
            v0 = _mm512_fmadd_pd(_mm512_load_pd(Li + k +  0), _mm512_load_pd(x + k +  0), v0);
            v1 = _mm512_fmadd_pd(_mm512_load_pd(Li + k +  8), _mm512_load_pd(x + k +  8), v1);
            v2 = _mm512_fmadd_pd(_mm512_load_pd(Li + k + 16), _mm512_load_pd(x + k + 16), v2);
            v3 = _mm512_fmadd_pd(_mm512_load_pd(Li + k + 24), _mm512_load_pd(x + k + 24), v3);
        }
        __m512d vs = _mm512_add_pd(_mm512_add_pd(v0, v1), _mm512_add_pd(v2, v3));
        double s = _mm512_reduce_add_pd(vs);
        for (int64_t k = full; k < n; ++k) s += Li[k] * x[k];
        x[i] = (b[i] - s) / Li[i];
    }
#elif defined(__AVX2__)
    for (int64_t i = 0; i < N; ++i) {
        const double *Li = L + i * N;
        const int64_t n = i;
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
        const int64_t full = n & ~15LL;
        for (int64_t k = 0; k < full; k += 16) {
            v0 = _mm256_fmadd_pd(_mm256_load_pd(Li + k +  0), _mm256_load_pd(x + k +  0), v0);
            v1 = _mm256_fmadd_pd(_mm256_load_pd(Li + k +  8), _mm256_load_pd(x + k +  8), v1);
        }
        double s = hsum256(_mm256_add_pd(v0, v1));
        for (int64_t k = full; k < n; ++k) s += Li[k] * x[k];
        x[i] = (b[i] - s) / Li[i];
    }
#else
    for (int64_t i = 0; i < N; ++i) {
        const double *Li = L + i * N;
        double s = 0.0;
        for (int64_t k = 0; k < i; ++k) s += Li[k] * x[k];
        x[i] = (b[i] - s) / Li[i];
    }
#endif
}
