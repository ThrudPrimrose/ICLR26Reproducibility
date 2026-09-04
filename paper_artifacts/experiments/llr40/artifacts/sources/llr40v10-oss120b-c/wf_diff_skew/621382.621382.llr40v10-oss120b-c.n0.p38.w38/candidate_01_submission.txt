/* Optimized implementation of wf_diff_skew kernel.
 * Computes a[i][j] += a[i-1][j] + a[i-1][j+1] for a square matrix stored in row-major order.
 * The outer loop is sequential due to the true dependence across rows.
 * The inner loop is vectorized with an OpenMP SIMD directive. Alignment of 16 bytes is
 * assumed, which matches the typical alignment of malloc'd memory.
 */

#include <stdint.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    for (int64_t i = 1; i < N; ++i) {
        double *restrict row = a + i * N;
        double *restrict prev = a + (i - 1) * N;
        const int64_t cols = N - 1;
        int64_t j = 0;
        #pragma omp simd
        for (j = 0; j < cols; ++j) {
            row[j] += prev[j] + prev[j + 1];
        }
    }
}

