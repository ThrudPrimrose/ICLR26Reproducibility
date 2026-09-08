/*
 * Parallel prefix sum (scan) for TSVC tsvc_2 's3112' kernel.
 * This version uses long double for the intermediate block sums
 * and offsets to reduce rounding error and match the reference
 * serial behavior within the allowed tolerance.
 */

#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D)
{
    if (LEN_1D <= 0)
        return;

    int nthreads = omp_get_max_threads();
    long double *block_sum = (long double *)malloc(sizeof(long double) * nthreads);
    if (block_sum == NULL) {
        // Fallback to serial execution on allocation failure.
        double sum = 0.0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            sum += a[i];
            b[i] = sum;
        }
        return;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t chunk = (LEN_1D + nthreads - 1) / nthreads;
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;

        /* First pass: compute the sum of this block using long double precision. */
        long double local_sum = 0.0L;
        for (int64_t i = start; i < end; ++i) {
            local_sum += (long double)a[i];
        }
        block_sum[tid] = local_sum;

        #pragma omp barrier

        /* Compute offset as the sum of all preceding block sums (long double). */
        long double offset = 0.0L;
        for (int i = 0; i < tid; ++i) {
            offset += block_sum[i];
        }

        /* Second pass: compute the prefix using the offset and store results as double. */
        long double acc = offset;
        for (int64_t i = start; i < end; ++i) {
            acc += (long double)a[i];
            b[i] = (double)acc;
        }
    }

    free(block_sum);
}
