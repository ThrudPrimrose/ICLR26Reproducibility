/* Host-only optimized implementation of fuse_move_ifs_fp64.
 * Uses OpenMP parallel loops without target offload to avoid lengthy device
 * compilation. The algorithm fuses the two original loops and processes the
 * i‑j matrix in a single collapsed parallel region. This retains correctness
 * while providing multithreaded speedup on the host.
 */

#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b,
                        const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
    if (K > 0) {
        /* Both a and b are written. */
        #pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            for (int64_t j = 0; j < LEN_2D; ++j) {
                const size_t idx = (size_t)i * (size_t)LEN_2D + (size_t)j;
                const double s = src[idx];
                b[idx] = s + 1.0;
                if (cond[i] > 0.0) {
                    a[idx] = s * 2.0;
                }
            }
        }
    } else {
        /* K <= 0: only a may be written. */
        #pragma omp parallel for collapse(2)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            for (int64_t j = 0; j < LEN_2D; ++j) {
                const size_t idx = (size_t)i * (size_t)LEN_2D + (size_t)j;
                const double s = src[idx];
                if (cond[i] > 0.0) {
                    a[idx] = s * 2.0;
                }
            }
        }
    }
}

