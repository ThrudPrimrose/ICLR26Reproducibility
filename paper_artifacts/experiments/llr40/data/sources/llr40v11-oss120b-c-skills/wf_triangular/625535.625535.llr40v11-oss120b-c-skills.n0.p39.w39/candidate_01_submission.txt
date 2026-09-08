/* Optimized wavefront triangular kernel using OpenMP parallelism.
 * Signature matches the reference: void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D)
 * The computation is: a[i,j] += a[i-1,j] + a[i,j-1] for j >= i, i >= 1.
 * Parallelized across anti-diagonals (i+j = k). Each diagonal can be computed in parallel
 * because dependencies are only on the previous diagonal.
 */

#include <stdint.h>
#include <omp.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    if (N <= 1) return; // nothing to do
    const int64_t k_min = 2;
    const int64_t k_max = 2 * (N - 1);
    #pragma omp parallel
    {
        for (int64_t k = k_min; k <= k_max; ++k) {
            int64_t i_start = k - (N - 1);
            if (i_start < 1) i_start = 1;
            int64_t i_end = k / 2;
            if (i_end > N - 1) i_end = N - 1;
            #pragma omp for schedule(static)
            for (int64_t i = i_start; i <= i_end; ++i) {
                int64_t j = k - i;
                double term = a[i * N + j] + a[(i - 1) * N + j] + a[i * N + (j - 1)];
                a[i * N + j] = term;
            }
        }
    }
}
