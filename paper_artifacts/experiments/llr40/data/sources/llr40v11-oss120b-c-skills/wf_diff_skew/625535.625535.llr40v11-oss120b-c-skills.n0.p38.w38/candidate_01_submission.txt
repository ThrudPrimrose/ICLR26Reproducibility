/* Optimized implementation of wf_diff_skew kernel.
 * Signature matches the reference: void wf_diff_skew_fp64(double *restrict a,
 *   const int64_t LEN_2D)
 *
 * The kernel updates a 2D square array stored in row-major order:
 *   a[i][j] = a[i][j] + a[i-1][j] + a[i-1][j+1]
 * for i = 1..LEN_2D-1 and j = 0..LEN_2D-2.
 *
 * The outer loop carries a true dependence across rows, therefore it must remain
 * sequential.  The inner loop has no inter-iteration dependence and can be
 * vectorised and parallelised with OpenMP SIMD.  The implementation uses pointer
 * arithmetic to remove repeated index calculations and hints the compiler to
 * vectorise with `#pragma omp simd`.
 */

#include <stdint.h>
#include <omp.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    // Guard against degenerate dimensions.
    if (LEN_2D <= 1)
        return;

    // Process rows sequentially because each row depends on the fully updated
    // previous row.
    for (int64_t i = 1; i < LEN_2D; ++i) {
        double *restrict row_i   = a + i * LEN_2D;
        double *restrict row_im1 = a + (i - 1) * LEN_2D;

        // The inner loop is independent across j; vectorise it.
        // LEN_2D-1 is the exclusive bound.
        #pragma omp simd
        for (int64_t j = 0; j < LEN_2D - 1; ++j) {
            row_i[j] += row_im1[j] + row_im1[j + 1];
        }
    }
}
