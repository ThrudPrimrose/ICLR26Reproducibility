#include <stdint.h>
#include <omp.h>

void wavefront2d_fp64(double *restrict a, int64_t LEN_2D, uint8_t *restrict workspace, int64_t workspace_bytes) {
    const int64_t N = LEN_2D;
    // The computation updates interior points (i,j) where i,j >= 1 and < N.
    // Use wavefront (anti-diagonal) order: all points with the same i+j can be processed in parallel.
    #pragma omp parallel
    {
        for (int64_t s = 2; s <= 2 * (N - 1); ++s) {
            // Determine the range of i for this diagonal.
            int64_t i_start = s - (N - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = s - 1;
            if (i_end > N - 1) i_end = N - 1;
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = s - i;
                // Compute the flat index for a[i][j].
                int64_t idx = i * N + j;
                // Load the four values needed for the update.
                double a_ij = a[idx];
                double a_im1_j = a[(i - 1) * N + j];
                double a_i_jm1 = a[i * N + (j - 1)];
                double a_im1_jm1 = a[(i - 1) * N + (j - 1)];
                a[idx] = 0.25 * (a_ij + a_im1_j + a_i_jm1 + a_im1_jm1);
            }
        }
    }
}

