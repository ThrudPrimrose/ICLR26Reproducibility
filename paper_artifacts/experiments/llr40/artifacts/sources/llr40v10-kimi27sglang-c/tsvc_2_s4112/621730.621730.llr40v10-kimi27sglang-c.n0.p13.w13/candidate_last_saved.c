#include <stdint.h>
#include <immintrin.h>

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    const __m512d two = _mm512_set1_pd(2.0);
    const int64_t n = LEN_1D & ~31;
    int64_t i;
    #pragma omp parallel for schedule(static) num_threads(24)
    for (i = 0; i < n; i += 32) {
        __m256i idx0 = _mm256_loadu_si256((const __m256i *)(ip + i));
        __m256i idx1 = _mm256_loadu_si256((const __m256i *)(ip + i + 8));
        __m256i idx2 = _mm256_loadu_si256((const __m256i *)(ip + i + 16));
        __m256i idx3 = _mm256_loadu_si256((const __m256i *)(ip + i + 24));
        __m512d bv0 = _mm512_i32gather_pd(idx0, b, 8);
        __m512d bv1 = _mm512_i32gather_pd(idx1, b, 8);
        __m512d bv2 = _mm512_i32gather_pd(idx2, b, 8);
        __m512d bv3 = _mm512_i32gather_pd(idx3, b, 8);
        __m512d av0 = _mm512_loadu_pd(a + i);
        __m512d av1 = _mm512_loadu_pd(a + i + 8);
        __m512d av2 = _mm512_loadu_pd(a + i + 16);
        __m512d av3 = _mm512_loadu_pd(a + i + 24);
        av0 = _mm512_fmadd_pd(bv0, two, av0);
        av1 = _mm512_fmadd_pd(bv1, two, av1);
        av2 = _mm512_fmadd_pd(bv2, two, av2);
        av3 = _mm512_fmadd_pd(bv3, two, av3);
        _mm512_storeu_pd(a + i, av0);
        _mm512_storeu_pd(a + i + 8, av1);
        _mm512_storeu_pd(a + i + 16, av2);
        _mm512_storeu_pd(a + i + 24, av3);
    }
    for (i = n; i < LEN_1D; ++i) {
        a[i] += b[ip[i]] * 2.0;
    }
}
