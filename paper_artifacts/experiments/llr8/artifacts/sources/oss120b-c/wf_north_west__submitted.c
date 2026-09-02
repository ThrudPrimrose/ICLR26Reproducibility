#include <stdint.h>
#include <omp.h>

// Wavefront north-west kernel: a[i][j] = a[i][j] + a[i-1][j] + a[i][j-1]
// a is a flattened 2D array stored in row-major order.
// LEN_2D is the dimension of the square matrix.
// The kernel updates the array in-place.

void wf_north_west_fp64(double *a, int64_t LEN_2D, uint8_t *workspace, int64_t workspace_bytes) {
    // Parallel wavefront implementation using a single parallel region.
    const int64_t max_sum = 2 * (LEN_2D - 1);
    #pragma omp parallel
    {
        for (int64_t sum = 2; sum <= max_sum; ++sum) {
            // Determine the range of i for this diagonal.
            int64_t i_start = sum > (LEN_2D - 1) ? sum - (LEN_2D - 1) : 1;
            int64_t i_end   = (sum - 1) < (LEN_2D - 1) ? (sum - 1) : (LEN_2D - 1);
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = sum - i;
                // Compute linear indices (row-major layout).
                int64_t idx      = i * LEN_2D + j;
                int64_t idx_up   = (i - 1) * LEN_2D + j;
                int64_t idx_left = i * LEN_2D + (j - 1);
                double val = a[idx];
                val = val + a[idx_up];
                val = val + a[idx_left];
                a[idx] = val;
            }
        }
    }
}
