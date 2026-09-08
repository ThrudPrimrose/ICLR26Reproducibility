#include <stdint.h>
#include <immintrin.h>

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    const int64_t vec_width = 4; // AVX2 processes 4 doubles per iteration.
    const __m256d two = _mm256_set1_pd(2.0);
    int64_t vec_end = (LEN_1D / vec_width) * vec_width; // Largest multiple of vec_width <= LEN_1D
    int64_t i;
    #pragma omp parallel for schedule(static) private(i)
    for (i = 0; i < vec_end; i += vec_width) {
        // Load current values of a (unaligned)
        __m256d a_vec = _mm256_loadu_pd(&a[i]);
        // Gather b indexed by ip (int32 indices) -> double values
        __m128i idx = _mm_loadu_si128((const __m128i *)&ip[i]); // loads 4 int32 indices
        __m256d b_vec = _mm256_i32gather_pd(b, idx, 8);
        // Fused multiply-add: a += 2.0 * b
        a_vec = _mm256_fmadd_pd(two, b_vec, a_vec);
        // Store result back to a
        _mm256_storeu_pd(&a[i], a_vec);
    }
    // Handle any remaining elements (if LEN_1D is not a multiple of vec_width)
    for (i = vec_end; i < LEN_1D; ++i) {
        a[i] += b[ip[i]] * 2.0;
    }
}
