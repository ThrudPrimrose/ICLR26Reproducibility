/* Optimized C implementation for the wf_diff_skew kernel.
 * Computes a[i,j] += a[i-1,j] + a[i-1,j+1] for a square matrix.
 * Parallelizes the inner j-loop using OpenMP and enables SIMD vectorization.
 */

#include <stdint.h>

/* The function signature follows the C-ABI expected by the benchmark. */
void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    /* Parallelize over columns (j) for each row (i). The outer loop over rows
     * must remain sequential due to the data dependency on the previous row. */
    #pragma omp parallel
    {
        for (int64_t i = 1; i < LEN_2D; ++i) {
            double *restrict cur = a + i * LEN_2D;
            double *restrict prev = a + (i - 1) * LEN_2D;
            /* Distribute j-iteration chunks among threads and let the compiler SIMD. */
            #pragma omp for simd schedule(static)
            for (int64_t j = 0; j < LEN_2D - 1; ++j) {
                cur[j] = cur[j] + prev[j] + prev[j + 1];
            }
        }
    }
}
