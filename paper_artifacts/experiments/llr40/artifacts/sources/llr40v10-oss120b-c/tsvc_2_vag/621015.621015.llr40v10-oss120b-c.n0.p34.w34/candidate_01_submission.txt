#include <stdint.h>
#include <x86intrin.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    // Vectorized gather using AVX2 for groups of 4 elements.
    int64_t vec_iters = LEN_1D / 4;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < vec_iters; ++i) {
        int64_t base = i * 4;
        __m128i idx = _mm_loadu_si128((const __m128i *)&ip[base]);
        __m256d vals = _mm256_i32gather_pd(b, idx, 8);
        _mm256_storeu_pd(&a[base], vals);
    }
    // Tail handling for any remaining elements.
    for (int64_t i = vec_iters * 4; i < LEN_1D; ++i) {
        a[i] = b[ip[i]];
    }
}
