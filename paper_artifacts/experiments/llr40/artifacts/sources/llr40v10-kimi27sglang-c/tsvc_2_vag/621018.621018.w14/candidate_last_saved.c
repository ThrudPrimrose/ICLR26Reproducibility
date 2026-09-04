#include <stdint.h>
#include <immintrin.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b,
                     const int32_t *restrict ip, const int64_t LEN_1D) {
    const int64_t n8 = LEN_1D & ~((int64_t)7);

    #pragma omp parallel for schedule(static) if(n8 > 4096)
    for (int64_t i = 0; i < n8; i += 8) {
        __m256i idx = _mm256_loadu_si256((const __m256i *)(ip + i));
        __m512d v = _mm512_i32gather_pd(idx, b, 8);
        _mm512_storeu_pd(a + i, v);
    }

    for (int64_t i = n8; i < LEN_1D; ++i) {
        a[i] = b[ip[i]];
    }
}
