/* Optimized implementation of TSVC tsvc_2 kernel s119 (wavefront).
 * Signature matches reference: void tsvc_2_s119_fp64(double *restrict aa,
 *                                           const double *restrict bb,
 *                                           const int64_t LEN_2D);
 * This version computes the updates along anti-diagonals and parallelizes each diagonal.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    if (N <= 1) return; // nothing to do
    // Parallel region: each anti-diagonal is processed sequentially, but its elements are independent.
    #pragma omp parallel
    {
        // loop over sum = i + j, which ranges from 2 to 2*(N-1).
        for (int64_t s = 2; s <= 2 * (N - 1); ++s) {
            // compute bounds for i on this diagonal: i in [max(1, s-(N-1)), min(N-1, s-1)]
            int64_t i_start = s - (N - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = s - 1;
            if (i_end > N - 1) i_end = N - 1;
            // parallelize over i for this diagonal. Implicit barrier after the for ensures correctness.
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = s - i;
                // idx of current element
                int64_t idx_ij = i * N + j;
                // idx of element from previous diagonal (i-1, j-1)
                int64_t idx_im1j = (i - 1) * N + (j - 1);
                aa[idx_ij] = aa[idx_im1j] + bb[idx_ij];
            }
        }
    }
}

