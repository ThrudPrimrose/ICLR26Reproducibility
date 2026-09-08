/* Wavefront triangular kernel optimized with OpenMP parallelism.
 * Original reference computes:
 *   a[i][j] += a[i-1][j] + a[i][j-1] for i=1..N-1, j=i..N-1
 * The computation has dependencies only on the previous anti-diagonal (i+j-1),
 * thus all points on a given anti-diagonal (i+j = t) can be processed in parallel.
 * This implementation iterates over anti-diagonals sequentially (t) and uses a
 * parallel for over the independent points on each diagonal.
 *
 * The loop bounds are derived as follows:
 *   For a given t, i ranges from max(1, t-(N-1)) to min(N-1, t/2).
 *   j is then t - i, guaranteeing j >= i and both indices within [0, N-1].
 */

#include <stdint.h>
#include <omp.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    if (LEN_2D <= 1) return; // nothing to do

    const int64_t max_t = 2 * LEN_2D - 2; // maximum i+j value (both indices 0‑based, i>=1)

#pragma omp parallel
    {
        for (int64_t t = 2; t <= max_t; ++t) {
            /* Determine the valid i range for this anti-diagonal.
               i must satisfy: 1 <= i <= LEN_2D-1,
               i <= j (= t-i)   => i <= t/2,
               j <= LEN_2D-1    => i >= t-(LEN_2D-1).
            */
            int64_t i_start = t - (LEN_2D - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = t / 2; // floor(t/2)
            if (i_end > LEN_2D - 1) i_end = LEN_2D - 1;
            if (i_start > i_end) continue; // no work on this diagonal

#pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = t - i;
                // Compute flat indices for readability.
                int64_t idx = i * LEN_2D + j;
                int64_t idx_up = (i - 1) * LEN_2D + j;
                int64_t idx_left = i * LEN_2D + (j - 1);
                a[idx] = a[idx] + a[idx_up] + a[idx_left];
            }
        }
    }
}

