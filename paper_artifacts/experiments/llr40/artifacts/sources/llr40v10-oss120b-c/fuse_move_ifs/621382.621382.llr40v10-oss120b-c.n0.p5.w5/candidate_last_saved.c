/* Optimized implementation of fuse_move_ifs kernel.
 * Performs conditional scaling of array a and unconditional offset addition to array b.
 * Uses OpenMP parallelism and SIMD vectorization, merging loops when possible for better cache reuse.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b,
                        const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
    // If K > 0 we can compute b for all elements while conditionally computing a.
    if (K > 0) {
        /* Parallelize over rows. Each iteration works on one row and thus avoids false sharing.
         * The inner loop is vectorized with OpenMP SIMD. Cond[i] is evaluated once per row.
         */
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            bool cond_true = cond[i] > 0.0;
            const double *src_row = src + i * LEN_2D;
            double *a_row = a + i * LEN_2D;
            double *b_row = b + i * LEN_2D;
            #pragma omp simd
            for (int64_t j = 0; j < LEN_2D; ++j) {
                double s = src_row[j];
                if (cond_true) {
                    a_row[j] = s * 2.0;
                }
                b_row[j] = s + 1.0;
            }
        }
    } else {
        // K <= 0: only compute a where cond[i] > 0.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            if (cond[i] > 0.0) {
                const double *src_row = src + i * LEN_2D;
                double *a_row = a + i * LEN_2D;
                #pragma omp simd
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    a_row[j] = src_row[j] * 2.0;
                }
            }
        }
    }
}

