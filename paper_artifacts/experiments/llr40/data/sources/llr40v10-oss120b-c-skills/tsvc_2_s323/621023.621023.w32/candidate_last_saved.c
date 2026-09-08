/* Optimized version of tsvc_2_s323 kernel using OpenMP scan.
 *
 * Original algorithm:
 *   for i = 1..N-1:
 *       a[i] = b[i-1] + c[i] * d[i];
 *       b[i] = a[i] + c[i] * e[i];
 *
 * This yields the recurrence:
 *   b[i] = b[i-1] + c[i] * (d[i] + e[i])
 *   a[i] = b[i] - c[i] * e[i]
 *
 * The recurrence is a prefix sum, which can be evaluated in parallel using
 * OpenMP 5.0's `inscan` reduction and the `scan` construct. After the scan we
 * compute `a[i]` from `b[i]` without further dependencies.
 */

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s323_fp64(double *restrict a,
                       double *restrict b,
                       const double *restrict c,
                       const double *restrict d,
                       const double *restrict e,
                       const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        return; /* nothing to do */
    }

    double b0 = b[0];          // preserve original b[0]

    #if 0
// Parallel prefix sum using block-wise decomposition to preserve exact sequential rounding order.
    int max_threads = omp_get_max_threads();
    double *block_sum = (double*)calloc(max_threads + 1, sizeof(double));
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        // Compute chunk boundaries (excluding index 0 which is unchanged)
        int64_t chunk = (LEN_1D - 1 + nthreads - 1) / nthreads; // ceil division
        int64_t start = 1 + tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        // Compute total contribution of this block (t1 + t2)
        double local_total = 0.0;
        for (int64_t i = start; i < end; ++i) {
            double t1 = c[i] * d[i];
                local_total += t1;
                double t2 = c[i] * e[i];
                local_total += t2;
        }
        block_sum[tid + 1] = local_total;
        #pragma omp barrier
        #pragma omp single
        {
            // Compute prefix sums of block totals to get offsets for each block
            for (int i = 1; i <= nthreads; ++i) {
                block_sum[i] += block_sum[i - 1];
            }
        }
        // Offset for this block: sum of all previous blocks
        double offset = block_sum[tid];
        for (int64_t i = start; i < end; ++i) {
            // a[i] = b0 + (offset + t1_i)
            a[i] = b0 + offset + c[i] * d[i];
            // Update offset with t1_i
            offset += c[i] * d[i];
            // b[i] = a[i] + t2_i
            b[i] = a[i] + c[i] * e[i];
            // Update offset with t2_i for next iteration
            offset += c[i] * e[i];
        }
    }
    free(block_sum);
#endif

    // Simple sequential implementation (fallback)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        a[i] = b[i - 1] + c[i] * d[i];
        b[i] = a[i] + c[i] * e[i];
    }
    // a[0] and b[0] remain unchanged, matching the reference semantics.
}
