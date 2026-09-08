#include <stdint.h>
#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>

/* b = src + 1 over one row, NT stores for the 64B-aligned middle. */
static inline void stream_b(const double *restrict s, double *restrict pb, const int64_t L) {
    const int64_t j = ((64 - (int64_t)((uintptr_t)pb & 63)) & 63) >> 3; /* first 64B-aligned index, 0..7 */
    const int64_t n1 = L & ~7;
    const int64_t jmin = j < L ? j : L;
    for (int64_t k = 0; k < jmin; ++k) pb[k] = s[k] + 1.0;
    const __m512d one = _mm512_set1_pd(1.0);
    for (int64_t k = j; k < n1; k += 8)
        _mm512_stream_pd(pb + k, _mm512_add_pd(_mm512_loadu_pd(s + k), one));
    for (int64_t k = n1; k < L; ++k) pb[k] = s[k] + 1.0;
}

/* a = 2*src over one row, NT stores for the 64B-aligned middle. */
static inline void stream_a(const double *restrict s, double *restrict pa, const int64_t L) {
    const int64_t j = ((64 - (int64_t)((uintptr_t)pa & 63)) & 63) >> 3;
    const int64_t n1 = L & ~7;
    const int64_t jmin = j < L ? j : L;
    for (int64_t k = 0; k < jmin; ++k) pa[k] = s[k] * 2.0;
    const __m512d two = _mm512_set1_pd(2.0);
    for (int64_t k = j; k < n1; k += 8)
        _mm512_stream_pd(pa + k, _mm512_mul_pd(_mm512_loadu_pd(s + k), two));
    for (int64_t k = n1; k < L; ++k) pa[k] = s[k] * 2.0;
}
#endif

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
#if defined(__AVX512F__)
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const double *restrict s = src + i * LEN_2D;
            if (K > 0) {
                double *restrict pb = b + i * LEN_2D;
                stream_b(s, pb, LEN_2D);
            }
            if (cond[i] > 0.0) {
                double *restrict pa = a + i * LEN_2D;
                stream_a(s, pa, LEN_2D);
            }
        }
        _mm_sfence(); /* land all NT stores before leaving the region */
    }
#else
    if (K > 0) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const double *restrict s = src + i * LEN_2D;
            double *restrict pb = b + i * LEN_2D;
            #pragma omp simd
            for (int64_t j = 0; j < LEN_2D; ++j) pb[j] = s[j] + 1.0;
        }
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        if (cond[i] > 0.0) {
            const double *restrict s = src + i * LEN_2D;
            double *restrict pa = a + i * LEN_2D;
            #pragma omp simd
            for (int64_t j = 0; j < LEN_2D; ++j) pa[j] = s[j] * 2.0;
        }
    }
#endif
}
