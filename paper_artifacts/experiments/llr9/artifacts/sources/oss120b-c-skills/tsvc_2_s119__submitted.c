#include <stdint.h>
#include <omp.h>

// TSVC tsvc_2 kernel s119
// Performs: aa[i,j] = aa[i-1,j-1] + bb[i,j]
// LEN_2D: dimension of square arrays (row-major order)
// All arrays are assumed to be of type double.
// The kernel uses a wavefront (diagonal) parallelization to respect the \n// dependence on the previous diagonal.
#include <string.h>
#include <stdlib.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    #pragma omp parallel default(none) shared(aa, bb, LEN_2D)
    {
        for (int64_t sum = 2; sum <= 2 * (LEN_2D - 1); ++sum) {
            int64_t i_start = sum - (LEN_2D - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = sum - 1;
            if (i_end > LEN_2D - 1) i_end = LEN_2D - 1;
            #pragma omp for schedule(static) nowait
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = sum - i;
                int64_t idx = i * LEN_2D + j;
                int64_t idx_up = (i - 1) * LEN_2D + (j - 1);
                aa[idx] = aa[idx_up] + bb[idx];
            }
            #pragma omp barrier
        }
    }
}

