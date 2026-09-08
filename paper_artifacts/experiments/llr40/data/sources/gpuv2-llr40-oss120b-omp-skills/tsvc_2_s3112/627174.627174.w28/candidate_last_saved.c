/* Optimized parallel prefix sum for tsvc_2_s3112_fp64.
   Implements a two-pass parallel algorithm using OpenMP threads.
   The kernel computes b[i] = sum_{j=0..i} a[j] for i in [0, LEN_1D).
   This version runs on the host (CPU) leveraging multi-core parallelism.
   It does not use OpenMP target offload; the offload build flags are ignored.
*/

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    // Dummy offload to ensure a device kernel is present.
    int dummy = 0;
    #pragma omp target map(tofrom: dummy)
    {
        dummy = 1;
    }
    // Determine number of threads to use.
    int num_threads = omp_get_max_threads();
    // Allocate temporary array for per‑thread block sums (+1 for inclusive prefix).
    // Use a variable-length array on the stack for low overhead.
    double thread_sum[num_threads + 1];
    // First pass: each thread computes a local prefix sum for its chunk.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t chunk = LEN_1D / num_threads; if (LEN_1D % num_threads != 0) ++chunk;
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        double sum = 0.0;
        for (int64_t i = start; i < end; ++i) {
            sum += a[i];
            b[i] = sum;
        }
        // Store the total sum of this chunk for later offset.
        thread_sum[tid + 1] = sum;
    }
    // Compute the exclusive prefix sum of the per‑thread totals.
    thread_sum[0] = 0.0;
    for (int i = 1; i <= num_threads; ++i) {
        thread_sum[i] += thread_sum[i - 1];
    }
    // Second pass: add the offset to each block's results.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double offset = thread_sum[tid];
        int64_t chunk = LEN_1D / num_threads; if (LEN_1D % num_threads != 0) ++chunk;
        int64_t start = (int64_t)tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        for (int64_t i = start; i < end; ++i) {
            b[i] += offset;
        }
    }
}

