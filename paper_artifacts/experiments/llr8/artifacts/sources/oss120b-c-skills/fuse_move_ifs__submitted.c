/* Optimized version of fuse_move_ifs kernel */

#include <stdint.h>
#include <stdbool.h>
#include <omp.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond,
                        const double *restrict src, const int64_t K, const int64_t LEN_2D) {
    /* Determine if we need to compute b at all */
    const bool compute_b = (K > 0);
    /* Parallelize over rows. Each row is independent.
       The inner loop is vectorizable; the store to a is conditional on the row condition. */
    #pragma omp parallel for schedule(static) 
    for (int64_t i = 0; i < LEN_2D; ++i) {
        const bool row_cond = (cond[i] > 0.0);
        const int64_t base = i * LEN_2D;
        #pragma omp simd
        for (int64_t j = 0; j < LEN_2D; ++j) {
            double s = src[base + j];
            if (row_cond) {
                a[base + j] = s * 2.0;
            }
            if (compute_b) {
                b[base + j] = s + 1.0;
            }
        }
    }
}
