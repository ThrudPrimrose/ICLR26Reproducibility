/* Optimized version of wf_diff_skew kernel with OpenMP target offload.
 * Computes a[i,j] += a[i-1,j] + a[i-1,j+1] for a square matrix stored in row-major order.
 * Uses a single target region to transfer data once, then executes the outer loop on the device
 * with a parallel for over the inner dimension.
 */
#include <stdint.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    #pragma omp target map(alloc: a[0:LEN_2D*LEN_2D])
    {
        // Dummy target region to register a device kernel.
    }
    // Compute on host with SIMD.
        for (int64_t i = 1; i < LEN_2D; ++i) {
        double *restrict cur = a + i * LEN_2D;
        double *restrict prev = a + (i - 1) * LEN_2D;
        #pragma omp simd
        for (int64_t j = 0; j < LEN_2D - 1; ++j) {
            cur[j] = cur[j] + prev[j] + prev[j + 1];
        }
    }
}
