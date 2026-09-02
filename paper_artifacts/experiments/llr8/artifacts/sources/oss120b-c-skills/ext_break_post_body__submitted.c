/*
 * ext_break_post_body kernel
 *
 * Implements the TSVC kernel "s482" where the loop body executes before a guard
 * that may break out of the loop. The reference behaviour (see the NumPy
 * implementation) is:
 *
 *   for (i = 0; i < LEN_1D; ++i) {
 *       a[i] = a[i] + b[i] * c[i];
 *       if (c[i] > b[i])
 *           break;
 *   }
 *
 * The function follows the signature expected by the benchmark harness:
 *   void ext_break_post_body(int64_t LEN_1D,
 *                           double *restrict a,
 *                           double *restrict b,
 *                           double *restrict c);
 *
 * The `restrict` qualifier matches the ABI guarantee that the input arrays do not
 * alias each other, which helps the compiler vectorise the loop. All loop
 * indices use `int64_t` as required by the coding style guidelines.
 */

#include <stdint.h>
#include <omp.h>

void ext_break_post_body_fp64(double *restrict a, double *restrict b, double *restrict c,
                         int64_t LEN_1D, uint8_t *restrict workspace, int64_t workspace_bytes)
{
    /* Two-pass implementation: first find break index, then vectorized compute. */
    int64_t break_idx = LEN_1D; // default: no break
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (c[i] > b[i]) {
            break_idx = i;
            break;
        }
    }
    int64_t effective_end = (break_idx < LEN_1D) ? break_idx + 1 : LEN_1D;
    (void)workspace; (void)workspace_bytes;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < effective_end; ++i) {
        a[i] = a[i] + b[i] * c[i];
    }
}
