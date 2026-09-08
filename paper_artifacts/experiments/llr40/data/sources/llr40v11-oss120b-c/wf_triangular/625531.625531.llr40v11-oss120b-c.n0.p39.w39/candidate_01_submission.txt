#include <stdint.h>

// Optimized wavefront triangular kernel using OpenMP parallelism over anti-diagonals.
// Computes a[i][j] += a[i-1][j] + a[i][j-1] for j >= i, i >= 1.
// a is a flat row-major 2D array of size LEN_2D x LEN_2D.
void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    // No work for N <= 1
    if (N <= 1) return;
    // Parallel region: each anti-diagonal can be processed in parallel across its elements.
    #pragma omp parallel
    {
        // The sum of indices i + j ranges from 2 (i=1,j=1) up to 2*N-2 (i=N-1,j=N-1).
        for (int64_t d = 2; d <= 2 * N - 2; ++d) {
            // Calculate the valid i range for this diagonal while keeping j = d - i.
            // i must be at least 1 and at most N-1, and j must be within [i, N-1].
            int64_t i_start = d - (N - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = d / 2; // floor(d/2) ensures i <= j
            if (i_end > N - 1) i_end = N - 1;
            // Parallel loop over i for this diagonal.
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = d - i;
                // Update element a[i][j].
                a[i * N + j] = a[i * N + j] + a[(i - 1) * N + j] + a[i * N + (j - 1)];
            }
        }
    }
}
