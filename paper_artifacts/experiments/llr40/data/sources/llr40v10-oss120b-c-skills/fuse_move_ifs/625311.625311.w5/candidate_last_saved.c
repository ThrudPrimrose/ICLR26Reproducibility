/* Optimized version of fuse_move_ifs for fp64.
 * Applies OpenMP parallelism and SIMD vectorization.
 */

#include <stdint.h>
#include <omp.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
    // Parallelize the row-wise conditional copy to 'a'.
    #pragma omp parallel for schedule(static) if(LEN_2D > 0)
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

    // Unconditional copy to 'b' when K > 0.
    if (K > 0) {
        #pragma omp parallel for schedule(static) if(LEN_2D > 0)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const double *src_row = src + i * LEN_2D;
            double *b_row = b + i * LEN_2D;
            #pragma omp simd
            for (int64_t j = 0; j < LEN_2D; ++j) {
                b_row[j] = src_row[j] + 1.0;
            }
        }
    }
}
