#include <stddef.h>
#include <stdint.h>
#include <omp.h>

// TSVC_2 kernel s119 implementation (double precision).
// Arrays are stored in row-major order, size LEN_2D x LEN_2D.
// aa is read/write, bb is read-only.
// The update follows: aa[i,j] = aa[i-1,j-1] + bb[i,j]
// for i,j = 1 .. LEN_2D-1 (0‑based indexing).
// The dependency is along the diagonal; we expose parallelism by processing
// anti‑diagonals (i+j = const) which are independent.

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, int64_t LEN_2D,
                       uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;            // Unused, required by the ABI.
    (void)workspace_bytes;     // Unused, required by the ABI.
    if (LEN_2D <= 1) return;

    // The maximum sum of indices for the anti‑diagonal (i+j) when both start at 1.
    const int64_t max_sum = 2 * (LEN_2D - 1);

    // Parallel region over anti‑diagonals.
    #pragma omp parallel
    {
        for (int64_t sum = 2; sum <= max_sum; ++sum) {
            // Determine the valid i range for this diagonal.
            int64_t i_start = sum - (LEN_2D - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = sum - 1;
            if (i_end > LEN_2D - 1) i_end = LEN_2D - 1;

            // The iterations of this loop are independent, so we can distribute them.
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = sum - i; // j is implicitly >=1 and <LEN_2D
                int64_t idx = i * LEN_2D + j;
                int64_t prev_idx = (i - 1) * LEN_2D + (j - 1);
                aa[idx] = aa[prev_idx] + bb[idx];
            }
        }
    }
}
