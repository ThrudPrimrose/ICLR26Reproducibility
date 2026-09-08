/* Optimized implementation for fuse_move_ifs kernel. */
#include <stdint.h>
#include <stdbool.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
    /* Fuse the two nested loops to improve cache locality and reduce memory traffic.
       Parallelize over rows (i) using OpenMP. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        bool compute_a = cond[i] > 0.0;
        for (int64_t j = 0; j < LEN_2D; ++j) {
            double src_val = src[i * LEN_2D + j];
            if (compute_a) {
                a[i * LEN_2D + j] = src_val * 2.0;
            }
            if (K > 0) {
                b[i * LEN_2D + j] = src_val + 1.0;
            }
        }
    }
}
