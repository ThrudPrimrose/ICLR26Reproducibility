#include <stdint.h>
#include <immintrin.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    int64_t imax = LEN_1D >= 16 ? LEN_1D - 15 : 0;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < imax; i += 16) {
        __m256i idx0 = _mm256_loadu_si256((const __m256i *)(ip + i));
        __m256i idx1 = _mm256_loadu_si256((const __m256i *)(ip + i + 8));
        __m512d v0 = _mm512_i32gather_pd(idx0, b, 8);
        __m512d v1 = _mm512_i32gather_pd(idx1, b, 8);
        _mm512_storeu_pd(a + i, v0);
        _mm512_storeu_pd(a + i + 8, v1);
    }
    int64_t tail_start = (LEN_1D / 16) * 16;
    for (int64_t i = tail_start; i < LEN_1D; ++i) {
        a[i] = b[ip[i]];
    }
}
