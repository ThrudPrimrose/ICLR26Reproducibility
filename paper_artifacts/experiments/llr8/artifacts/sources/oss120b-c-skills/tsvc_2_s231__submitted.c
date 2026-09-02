/*
 * TSVC kernel "s231" implementation in C.
 *
 * The reference Python implementation (see /shared/tasks/tsvc_2_s231/tsvc_2_s231_numpy.py)
 * performs a column-wise prefix sum:
 *
 *   for i in range(LEN_2D):
 *       for j in range(1, LEN_2D):
 *           aa[j,i] = aa[j-1,i] + bb[j,i]
 *
 * In C the arrays are stored row-major, so the element aa[j,i] is located at
 *   aa[j*LEN_2D + i]
 * and similarly for bb. Each column (fixed i) is independent, therefore the
 * outer loop over i can be parallelised with OpenMP. The inner loop carries a
 * true data dependence on the previous row, so it must remain sequential.
 */

#include <stdint.h>
#include <omp.h>

/*
 * Function signature follows the benchmark convention: the function name matches
 * the kernel key (the last path component) and the arguments are plain pointers
 * to the data followed by the size parameter. The arrays contain double-precision
 * floating‑point values. `restrict` informs the compiler that the pointers do
 * not alias, enabling better optimisation.
 */
void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes)
{
    /* Parallelise over columns (index i). Each thread works on a contiguous
       span of columns, providing good cache locality and avoiding false
       sharing. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        /* The first row (j = 0) is left unchanged – it already contains the
           initial value required for the recurrence. */
        for (int64_t j = 1; j < LEN_2D; ++j) {
            /* Compute linear indices for the current and previous rows.
               The stride between rows is LEN_2D because the innermost dimension
               (i) varies fastest in C's row-major layout. */
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i];
        }
    }
}
