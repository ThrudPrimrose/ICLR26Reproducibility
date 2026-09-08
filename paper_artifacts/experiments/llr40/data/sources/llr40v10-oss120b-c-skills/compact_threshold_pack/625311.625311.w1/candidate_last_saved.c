/* Compact Threshold Pack kernel: stream compaction with weight multiplication.
 * For each element of src, if src[i] > 0, compute src[i] * weight[i] and store
 * into packed preserving order. The total number of survivors is stored in
 * out_count[0].
 *
 * Parallel implementation using OpenMP: each thread counts survivors in its
 * assigned chunk, then a prefix sum over per‑thread counts yields the output
 * offset for each thread. Threads then write their survivors using the computed
 * offset. This avoids atomics on the hot path.
 */

#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

void compact_threshold_pack_fp64(const double *restrict src,
                                 const double *restrict weight,
                                 double *restrict packed,
                                 int64_t *restrict out_count,
                                 const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        out_count[0] = 0;
        return;
    }

    // Allocate per‑thread count array using the maximum number of threads.
    const int max_threads = omp_get_max_threads();
    int64_t *thread_counts = (int64_t *)calloc((size_t)max_threads, sizeof(int64_t));
    int64_t *thread_offsets = (int64_t *)malloc((size_t)max_threads * sizeof(int64_t));
    if (!thread_counts || !thread_offsets) {
        // Allocation failure – fall back to serial version.
        int64_t n = 0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                packed[n++] = src[i] * weight[i];
            }
        }
        out_count[0] = n;
        free(thread_counts);
        free(thread_offsets);
        return;
    }

    // First parallel region: count survivors per thread.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t local_cnt = 0;
        // Count survivors in a static schedule chunk.
        #pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                local_cnt++;
            }
        }
        thread_counts[tid] = local_cnt;
        #pragma omp barrier

        // Compute exclusive prefix sum of thread counts to obtain per‑thread offsets.
        #pragma omp single
        {
            int64_t sum = 0;
            // omp_get_num_threads() gives the number of threads actually used.
            int num_threads = omp_get_num_threads();
            for (int i = 0; i < num_threads; ++i) {
                int64_t cnt = thread_counts[i];
                thread_offsets[i] = sum;
                sum += cnt;
            }
            out_count[0] = sum; // total number of survivors.
        }
        #pragma omp barrier

        // Write survivors using the computed offset.
        int64_t offset = thread_offsets[tid];
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                packed[offset++] = src[i] * weight[i];
            }
        }
    }

    free(thread_counts);
    free(thread_offsets);
}

