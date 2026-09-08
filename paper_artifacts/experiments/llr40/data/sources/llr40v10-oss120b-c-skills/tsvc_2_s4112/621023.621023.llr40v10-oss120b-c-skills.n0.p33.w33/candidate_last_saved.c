#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    const __m256d two = _mm256_set1_pd(2.0);
    // Process in blocks of 4 using AVX2 gather
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= LEN_1D - 4; i += 4) {
        // Load 4 indices (int32) from ip
        __m128i idx = _mm_loadu_si128((const __m128i *)&ip[i]);
        // Gather 4 double values from b using these indices (scale = 8 bytes)
        __m256d b_vals = _mm256_i32gather_pd(b, idx, 8);
        // Multiply by 2.0
        __m256d prod = _mm256_mul_pd(b_vals, two);
        // Load current values of a
        __m256d a_vals = _mm256_loadu_pd(&a[i]);
        // Add
        __m256d res = _mm256_add_pd(a_vals, prod);
        // Store back to a
        _mm256_storeu_pd(&a[i], res);
    }
    // Remainder loop for any leftover elements
    for (int64_t i = (LEN_1D / 4) * 4; i < LEN_1D; ++i) {
        a[i] += b[ip[i]] * 2.0;
    }
}
