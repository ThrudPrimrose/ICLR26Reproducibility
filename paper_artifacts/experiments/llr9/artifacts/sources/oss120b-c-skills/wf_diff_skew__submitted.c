#include <stdint.h>
#include <omp.h>

/*
 * Wavefront difference kernel with skew.
 * In-place update of a 2D square array a (row-major layout).
 * a[i][j] = a[i][j] + a[i-1][j] + a[i-1][j+1]
 * for i = 1..LEN_2D-1, j = 0..LEN_2D-2.
 *
 * The reference uses double precision; the engine expects the entry point named
 * `wf_diff_skew_fp64` with the standard signature:
 *   void wf_diff_skew_fp64(double *restrict a,
 *                         int64_t LEN_2D,
 *                         uint8_t *restrict workspace,
 *                         int64_t workspace_bytes);
 * The workspace arguments are unused for this kernel but must be present.
 */

void wf_diff_skew_fp64(double *restrict a, int64_t LEN_2D, uint8_t *restrict workspace, int64_t workspace_bytes) {
    (void)workspace;        // Unused, silence compiler warnings.
    (void)workspace_bytes;  // Unused.

    // Parallelize the inner column loop; outer row loop is inherently sequential due
    // to the dependence on the previous row.
    #pragma omp parallel
    {
        for (int64_t i = 1; i < LEN_2D; ++i) {
            double *restrict a_i   = a + i * LEN_2D;
            double *restrict a_im1 = a + (i - 1) * LEN_2D;
            #pragma omp for simd schedule(static)
            for (int64_t j = 0; j < LEN_2D - 1; ++j) {
                a_i[j] = a_i[j] + a_im1[j] + a_im1[j + 1];
            }
        }
    }
}

