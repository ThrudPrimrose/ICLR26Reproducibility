/* Optimized version of fuse_move_ifs microkernel.
 * Original reference: /shared/tasks/fuse_move_ifs/fuse_move_ifs_reference.c
 *
 * The kernel computes two separate operations on a 2D square matrix.
 *   a[i,j] = src[i,j] * 2.0  if cond[i] > 0.0
 *   b[i,j] = src[i,j] + 1.0  if K > 0
 *
 * The reference implementation performs two separate loop nests, which reads src twice.
 * Here we fuse the loops when K > 0, performing a single read of src per element and
 * conditional store to a. When K <= 0 we skip the b writes entirely.
 *
 * Parallelism: we parallelise the outer i-loop with OpenMP. The inner j-loop is left to
 * the compiler for SIMD vectorisation; an explicit #pragma omp simd is added to aid it.
 *
 * The pointers are declared restrict to inform the compiler about lack of aliasing.
 */

#include <stdint.h>
#include <stdbool.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b,
                        const double *restrict cond,
                        const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
    if (K > 0) {
        /* K positive: compute both a (conditionally) and b together.
         * The outer loop is parallelised. The inner loop is SIMD‑vectorisable.
         */
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const double *restrict src_row = src + i * LEN_2D;
            double *restrict b_row = b + i * LEN_2D;
            double *restrict a_row = a + i * LEN_2D;
            bool write_a = cond[i] > 0.0; // loop‑invariant for this row

            #pragma omp simd
            for (int64_t j = 0; j < LEN_2D; ++j) {
                double v = src_row[j];
                b_row[j] = v + 1.0;
                if (write_a) {
                    a_row[j] = v * 2.0;
                }
            }
        }
    } else {
        /* K non‑positive: only the a‑updates are needed.
         * Separate parallel region avoids the conditional inside the inner loop.
         */
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            if (cond[i] > 0.0) {
                const double *restrict src_row = src + i * LEN_2D;
                double *restrict a_row = a + i * LEN_2D;

                #pragma omp simd
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    a_row[j] = src_row[j] * 2.0;
                }
            }
        }
    }
}
