#include <stdint.h>
#include <omp.h>

void wf_triangular_fp64(double * restrict a, int64_t LEN_2D, uint8_t * restrict workspace, int64_t workspace_bytes) {
    // Triangular wavefront: a[i,j] depends on a[i-1,j] and a[i,j-1]
    // Parallelize over anti-diagonals (i + j constant)
    const int64_t N = LEN_2D;
    // The smallest sum of indices (i+j) is 2 (i=1,j=1). The largest is 2*(N-1).
    #pragma omp parallel
    {
        for (int64_t s = 2; s <= 2 * (N - 1); ++s) {
            // Determine i range for this anti-diagonal respecting the triangular domain (j >= i).
            int64_t i_start = s - (N - 1);
            if (i_start < 1) i_start = 1; // i must be at least 1
            int64_t i_end = s / 2; // floor division ensures i <= j
            if (i_end > N - 1) i_end = N - 1;
            // Parallel work over independent points on this anti-diagonal.
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = s - i; // column index
                // Update cell (i,j)
                a[i * N + j] = a[i * N + j] + a[(i - 1) * N + j] + a[i * N + (j - 1)];
            }
        }
    }
}
