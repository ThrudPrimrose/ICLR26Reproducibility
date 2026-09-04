/* Optimized implementation of TSVC tsvc_2 kernel s231 (fp64).
 * Original reference loops over columns (i) outer and rows (j) inner,
 * causing strided memory accesses that hinder vectorization.
 * This version swaps the loops: outer over rows (j) and inner over columns (i),
 * giving contiguous accesses for both input and output arrays. The inner loop is
 * annotated with OpenMP SIMD for vectorization and executed inside a parallel
 * region with an OpenMP "for" to distribute columns across threads.
 *
 * The algorithm computes a column-wise prefix sum of aa, adding the corresponding
 * values from bb at each step:
 *   aa[j,i] = aa[j-1,i] + bb[j,i] for j = 1..N-1, i = 0..N-1.
 */

#include <stdint.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    // Parallel region to reuse the same team across rows.
    #pragma omp parallel
    {
        // Process each row sequentially because each row depends on the previous one.
        for (int64_t j = 1; j < LEN_2D; ++j) {
            const double *restrict bb_row = bb + j * LEN_2D;
            const double *restrict aa_prev = aa + (j - 1) * LEN_2D;
            double *restrict aa_curr = aa + j * LEN_2D;
            // Split the columns across threads and let the compiler vectorize.
            #pragma omp for nowait schedule(static)
            for (int64_t i = 0; i < LEN_2D; ++i) {
                aa_curr[i] = aa_prev[i] + bb_row[i];
            }
        }
    }
}
