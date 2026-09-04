/* Optimized implementation of wf_triangular kernel using wavefront parallelism.
 * The kernel computes:
 *   a[i][j] = a[i][j] + a[i-1][j] + a[i][j-1]
 * for 0 <= i,j < LEN_2D with j >= i and i >= 1.
 * The reference implementation loops over i then j sequentially. Here we rewrite
 * the computation in terms of anti-diagonal (i+j) wavefronts. Cells on the same
 * anti‑diagonal are independent because each depends only on the cell directly
 * above (i‑1,j) and to the left (i,j‑1), both of which lie on the previous
 * anti‑diagonal (i+j‑1). Therefore we can safely parallelise each diagonal with
 * OpenMP.
 */

#include <stdint.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    if (LEN_2D <= 1) {
        return; // nothing to do for trivial matrices
    }

    // No auxiliary row pointer array – compute indices directly.
    const int64_t max_sum = 2 * (LEN_2D - 1);

    #pragma omp parallel
    {
        for (int64_t d = 2; d <= max_sum; ++d) {
            int64_t i_start = d - (LEN_2D - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = d / 2;
            if (i_end > LEN_2D - 1) i_end = LEN_2D - 1;
            if (i_start > i_end) continue;
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = d - i;
                int64_t base_i = i * LEN_2D;
                int64_t base_up = base_i - LEN_2D; // (i-1)*LEN_2D
                int64_t idx = base_i + j;
                int64_t idx_left = idx - 1;
                a[idx] = a[idx] + a[base_up + j] + a[idx_left];
            }
        }
    }
}

