/* Optimized version of fuse_move_ifs kernel.
 * This implementation fuses the two loop nests when K > 0, reducing memory traffic.
 * It also uses OpenMP parallelism and SIMD directives for better vectorization.
 */

#include <stdint.h>
#include <stddef.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
    // Compute whether we need to compute b at all.
    const int doK = (K > 0);

    if (doK) {
        // Fuse the two loops: for each row, conditionally compute a and always compute b.
        // Parallelize over rows. Inner loop is vectorizable.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            const int cond_pos = (cond[i] > 0.0);
            int64_t row_offset = i * LEN_2D;
            // Use SIMD for the inner loop.
            #pragma omp simd
            for (int64_t j = 0; j < LEN_2D; ++j) {
                double src_val = src[row_offset + j];
                if (cond_pos) {
                    a[row_offset + j] = src_val * 2.0;
                }
                b[row_offset + j] = src_val + 1.0;
            }
        }
    } else {
        // K <= 0: only compute a where cond is positive.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            if (cond[i] > 0.0) {
                int64_t row_offset = i * LEN_2D;
                #pragma omp simd
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    a[row_offset + j] = src[row_offset + j] * 2.0;
                }
            }
        }
    }
}
