#include <stdint.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    // Wavefront parallelism across anti-diagonals (i + j = sum).
    // Each point (i, j) depends only on the previous diagonal (i-1, j-1).
    // Therefore points on the same diagonal can be processed concurrently.
    #pragma omp parallel
    {
        // sum = i + j, where i and j range from 1 to LEN_2D-1 inclusive.
        // Minimum sum is 2 (i=1,j=1), maximum is 2*LEN_2D-2 (i=LEN_2D-1, j=LEN_2D-1).
        for (int64_t sum = 2; sum < 2 * LEN_2D; ++sum) {
            // Compute the valid range of i for this sum.
            int64_t i_start = sum - (LEN_2D - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = sum - 1;
            if (i_end > LEN_2D - 1) i_end = LEN_2D - 1;
            // Parallel loop over i (and consequently j) for this diagonal.
            #pragma omp for schedule(static) nowait
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = sum - i;
                int64_t idx = i * LEN_2D + j;
                int64_t idx_im1j = (i - 1) * LEN_2D + (j - 1);
                aa[idx] = aa[idx_im1j] + bb[idx];
            }
            // Synchronize before moving to the next diagonal, because the next diagonal
            // reads values written in this one.
            #pragma omp barrier
        }
    }
}
