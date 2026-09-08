/* Optimized version for tsvc_2_s119 using SIMD */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    for (int64_t i = 1; i < LEN_2D; ++i) {
        #pragma omp simd
        for (int64_t j = 1; j < LEN_2D; ++j) {
            const int64_t idx_ij = i * LEN_2D + j;               // [i][j]
            const int64_t idx_im1j = (i - 1) * LEN_2D + (j - 1); // [i-1][j-1]
            aa[idx_ij] = aa[idx_im1j] + bb[idx_ij];
        }
    }
}
