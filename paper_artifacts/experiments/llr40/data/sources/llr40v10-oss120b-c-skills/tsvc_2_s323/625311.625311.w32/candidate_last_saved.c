/* Optimized version of TSVC tsvc_2_s323 kernel.
 * Original reference (serial) computes a recurrence on b and writes a.
 * This implementation parallelizes the computation using a two-pass prefix sum.
 * It uses OpenMP for thread parallelism. */

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e,
                      const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        return;
    }
    const double base = b[0]; // initial value of b[0]
    const int64_t n = LEN_1D;

    // Allocate temporary array to hold per‑thread sums (size = thread count + 1).
    // We cannot know the exact thread count before the parallel region, so allocate a generous maximum.
    // omp_get_max_threads() is safe here because the parallel region will never exceed it.
    const int max_threads = omp_get_max_threads();
    double *thread_sums = (double *)aligned_alloc(64, (max_threads + 1) * sizeof(double));
    // Initialise to zero in case fewer threads are used.
    for (int i = 0; i <= max_threads; ++i) {
        thread_sums[i] = 0.0;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        // Work on the range [1, n) split among threads.
        int64_t total = n - 1;                    // number of elements to process
        int64_t chunk = (total + nth - 1) / nth;   // ceiling division
        int64_t start = 1 + tid * chunk;
        int64_t end = start + chunk;
        if (end > n) end = n;

        double sum = 0.0;
        for (int64_t i = start; i < end; ++i) {
            double contrib = c[i] * (d[i] + e[i]);
            sum += contrib;
            b[i] = sum; // store the local prefix sum (without base)
        }
        // Store this thread's total contribution; offset by +1 because thread_sums[0] will hold 0.
        thread_sums[tid + 1] = sum;

        #pragma omp barrier
        #pragma omp single
        {
            // Convert per‑thread totals into offsets.
            thread_sums[0] = 0.0;
            for (int i = 1; i <= nth; ++i) {
                thread_sums[i] += thread_sums[i - 1];
            }
        }
        // Apply the offset (including the original b[0]) to the local prefix sums.
        double offset = base + thread_sums[tid];
        for (int64_t i = start; i < end; ++i) {
            b[i] = offset + b[i];
        }
    }
    free(thread_sums);

    // Compute a using the final b values.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n; ++i) {
        a[i] = b[i - 1] + c[i] * d[i];
    }
}

