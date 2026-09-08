/* Optimized implementation of TSVC tsvc_2 s323 kernel.
 * Original reference computes:
 *   for i = 1..LEN_1D-1:
 *     a[i] = b[i-1] + c[i] * d[i];
 *     b[i] = a[i] + c[i] * e[i];
 *
 * This can be rewritten as a prefix-sum (scan) of the term
 *   temp[i] = c[i] * (d[i] + e[i]), i>=1
 * such that
 *   b[i] = b[0] + sum_{k=1..i} temp[k]
 * and then a[i] = b[i] - c[i] * e[i].
 * The implementation performs a parallel scan using a two-pass algorithm:
 *   1) each thread computes a local prefix sum for its chunk and stores the
 *      partial sums in b (temporarily) while recording the total sum of the
 *      chunk.
 *   2) a single thread computes the offsets for each thread based on the
 *      per‑thread totals and adds them to the locally stored prefix values.
 *   3) a final parallel loop computes a[i] from the completed b[i].
 *
 * This removes the loop‑carried dependence and enables OpenMP parallelism.
 */

#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e,
                      const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        /* No work to do – the original loop starts at i=1 */
        return;
    }

    const int64_t n = LEN_1D;
    const int max_threads = omp_get_max_threads();

    // Allocate per‑thread sum and offset arrays on the heap.
    double *thread_sum = (double *)malloc(max_threads * sizeof(double));
    double *thread_offset = (double *)malloc(max_threads * sizeof(double));
    if (!thread_sum || !thread_offset) {
        // Fallback to a sequential implementation if allocation fails.
        for (int64_t i = 1; i < n; ++i) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
        free(thread_sum);
        free(thread_offset);
        return;
    }

    // First pass: each thread computes a local prefix sum of temp[i] = c[i] * (d[i] + e[i])
    // and stores that temporary sum in b[i].
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        // Compute chunk boundaries for the range [1, n).
        int64_t chunk = (n - 1 + nt - 1) / nt; // ceil division
        int64_t i_start = 1 + (int64_t)tid * chunk;
        int64_t i_end = i_start + chunk;
        if (i_end > n) i_end = n;
        double sum = 0.0;
        for (int64_t i = i_start; i < i_end; ++i) {
            sum += c[i] * (d[i] + e[i]);
            b[i] = sum; // store local prefix value (without global offset)
        }
        thread_sum[tid] = sum;

        #pragma omp barrier

        // Single thread builds the offsets array.
        #pragma omp single
        {
            double offset = b[0]; // base value for b[0]
            for (int t = 0; t < nt; ++t) {
                thread_offset[t] = offset;
                offset += thread_sum[t];
            }
        }

        double off = thread_offset[tid];
        for (int64_t i = i_start; i < i_end; ++i) {
            b[i] = off + b[i]; // add the correct offset to obtain final b[i]
        }
    }

    // Second pass: compute a[i] from the now‑correct b[i].
#pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < n; ++i) {
        a[i] = b[i] - c[i] * e[i];
    }

    // Clean up.
    free(thread_sum);
    free(thread_offset);
}

