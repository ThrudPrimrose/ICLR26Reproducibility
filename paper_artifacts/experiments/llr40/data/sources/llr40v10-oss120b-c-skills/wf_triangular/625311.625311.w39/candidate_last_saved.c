/* Optimized wavefront triangular kernel using OpenMP parallelization across anti-diagonals.
 * Computes a[i][j] += a[i-1][j] + a[i][j-1] for j >= i, i in [1, LEN_2D-1].
 * The algorithm executes each anti-diagonal (i + j = s) sequentially, while parallelizing the
 * independent updates within a diagonal.
 */

#include <stdint.h>
#include <omp.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    // The total number of anti-diagonals to process (excluding the trivial outer boundary)
    const int64_t max_s = 2 * (LEN_2D - 1);
    // Parallel region; all threads cooperate on each diagonal.
    #pragma omp parallel
    {
        for (int64_t s = 2; s <= max_s; ++s) {
            // Determine the range of i for this diagonal:
            // i >= 1, i >= s - (LEN_2D - 1), i <= (LEN_2D - 1), i <= floor(s/2)
            int64_t i_start = 1;
            int64_t tmp = s - (LEN_2D - 1);
            if (tmp > i_start) i_start = tmp;
            int64_t i_end = (s >> 1); // floor(s/2)
            if (i_end > LEN_2D - 1) i_end = LEN_2D - 1;

            // Distribute the independent work of each diagonal across threads.
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = s - i;
                // Row pointers for quick address computation.
                double *row_i   = &a[i * LEN_2D];
                double *row_im1 = &a[(i - 1) * LEN_2D];
                // Update the element using values from the previous row and previous column.
                // Both reads are already computed on diagonal s-1.
                row_i[j] = row_i[j] + row_im1[j] + row_i[j - 1];
            }
            // Implicit barrier at end of omp for ensures all threads finish the diagonal.
        }
    }
}

