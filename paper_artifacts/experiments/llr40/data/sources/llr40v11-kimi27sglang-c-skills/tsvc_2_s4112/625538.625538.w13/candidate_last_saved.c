#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    const int64_t n16 = LEN_1D & ~15;
    const int64_t n8 = LEN_1D & ~7;
    const __m512d two = _mm512_set1_pd(2.0);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n16; i += 16) {
        __m256i idx0 = _mm256_loadu_si256((const __m256i *)&ip[i]);
        __m256i idx1 = _mm256_loadu_si256((const __m256i *)&ip[i + 8]);
        __m512d vb0 = _mm512_i32gather_pd(idx0, b, 8);
        __m512d vb1 = _mm512_i32gather_pd(idx1, b, 8);
        vb0 = _mm512_mul_pd(vb0, two);
        vb1 = _mm512_mul_pd(vb1, two);
        __m512d va0 = _mm512_loadu_pd(&a[i]);
        __m512d va1 = _mm512_loadu_pd(&a[i + 8]);
        va0 = _mm512_add_pd(va0, vb0);
        va1 = _mm512_add_pd(va1, vb1);
        _mm512_storeu_pd(&a[i], va0);
        _mm512_storeu_pd(&a[i + 8], va1);
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i = n16; i < n8; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i *)&ip[i]);
        __m512d vb = _mm512_i32gather_pd(idx, b, 8);
        vb = _mm512_mul_pd(vb, two);
        __m512d va = _mm512_loadu_pd(&a[i]);
        va = _mm512_add_pd(va, vb);
        _mm512_storeu_pd(&a[i], va);
    }
    for (int64_t i = n8; i < LEN_1D; ++i) {
        a[i] += b[ip[i]] * 2.0;
    }
}
