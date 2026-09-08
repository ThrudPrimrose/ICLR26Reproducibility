/* Optimized version of fuse_move_ifs for double precision.
 * Fuses the two separate loops into a single nested loop, reducing memory traffic.
 * Applies OpenMP parallelism on the outer dimension and SIMD vectorization on the inner.
 */

#include <stdint.h>
#include <omp.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b,
                        const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
    const int write_b = K > 0; // cache condition
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        const int write_a = (cond[i] > 0.0);
        // The inner loop is vectorizable. The scalar flags are loop‑invariant, enabling masked stores.
        #pragma omp simd
        for (int64_t j = 0; j < LEN_2D; ++j) {
            double s = src[i * LEN_2D + j];
            if (write_a) a[i * LEN_2D + j] = s * 2.0;
            if (write_b) b[i * LEN_2D + j] = s + 1.0;
        }
    }
}

