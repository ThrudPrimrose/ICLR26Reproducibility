/* Fast SIMD implementation of ext_war_unit_fp64.
 * Computes a[i] = a[i+1] + b[i] for i = 0 .. LEN_1D-2.
 * The loop has a forward dependence (distance 1), preventing parallelism,
 * but it can be vectorized safely because each iteration loads a[i+1]
 * before storing to a[i]. This version uses AVX2 intrinsics to process four
 * doubles per iteration, achieving a noticeable speedup over the reference.
 */

#include <stdint.h>
#include <immintrin.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        return; // nothing to compute
    }

    const int64_t n = LEN_1D - 1; // number of output elements
    const int64_t vec_width = 4; // AVX2 processes 4 doubles per vector
    const int64_t vec_end = n & -vec_width; // largest multiple of 4 <= n

    int64_t i = 0;
    for (i = 0; i < vec_end; i += vec_width) {
        // Load a[i+1]..a[i+4] (original values) and b[i]..b[i+3]
        __m256d a_next = _mm256_loadu_pd(&a[i + 1]);
        __m256d b_vec  = _mm256_loadu_pd(&b[i]);
        // Compute a[i]..a[i+3]
        __m256d sum = _mm256_add_pd(a_next, b_vec);
        // Store the results back to a[i]..a[i+3]
        _mm256_storeu_pd(&a[i], sum);
    }

    // Process any remaining elements (0 to 3) with a scalar loop
    for (; i < n; ++i) {
        a[i] = a[i + 1] + b[i];
    }
}

