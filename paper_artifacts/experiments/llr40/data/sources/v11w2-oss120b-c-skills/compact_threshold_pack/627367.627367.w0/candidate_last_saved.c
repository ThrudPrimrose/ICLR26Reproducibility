/*
 * Kernel: compact_threshold_pack_fp64
 * Stream compaction with per-element weighting.
 * Packs src[i] * weight[i] for all src[i] > 0.0 into packed array preserving order.
 * Stores the number of packed elements in out_count[0].
 *
 * The naive serial implementation is O(N) with a loop-carried index.
 * This implementation parallelizes the work using OpenMP.
 * It performs two passes:
 *   1) Each thread counts its survivors in its assigned chunk.
 *   2) A prefix sum of per-thread counts yields output offsets.
 *   3) Each thread writes its survivors into the final packed array using its offset.
 *
 * The algorithm uses only reads from src and weight, and writes to packed and out_count.
 * All pointers are marked restrict to aid alias analysis and vectorization.
 */

#include <stdint.h>
#include <stddef.h>
#include <omp.h>
#include <stdlib.h>

/*
 * void compact_threshold_pack_fp64(int64_t *restrict out_count,
 *                                 double *restrict src,
 *                                 double *restrict packed,
 *                                 double *restrict weight,
 *                                 const int64_t LEN_1D,
 *                                 uint8_t *restrict workspace,
 *                                 const int64_t workspace_len);
 */

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 double *restrict src,
                                 double *restrict weight,
                                 const int64_t LEN_1D,
                                 uint8_t *restrict workspace,
                                 const int64_t workspace_len) {
    if (LEN_1D <= 0) {
        if (out_count) out_count[0] = 0;
        return;
    }

    /* Determine the number of threads that will be used.
     * omp_get_max_threads may return a number larger than the actual parallelism
     * if the runtime decides otherwise, but we can safely allocate that many slots.
     */
    int nthreads = omp_get_max_threads();
    int64_t *thread_counts = (int64_t *)calloc((size_t)nthreads, sizeof(int64_t));
    int64_t *thread_offsets = (int64_t *)malloc((size_t)nthreads * sizeof(int64_t));
    if (!thread_counts || !thread_offsets) {
        /* Allocation failure – fall back to serial implementation */
        int64_t n = 0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                packed[n] = src[i] * weight[i];
                ++n;
            }
        }
        out_count[0] = n;
        free(thread_counts);
        free(thread_offsets);
        return;
    }

    /* First parallel region: count survivors per thread. */
    #pragma omp parallel num_threads(nthreads) default(none) shared(src, weight, LEN_1D, thread_counts, nthreads)
    {
        int tid = omp_get_thread_num();
        int64_t chunk = (LEN_1D + nthreads - 1) / nthreads; // ceil division
        int64_t start = tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        int64_t local_cnt = 0;
        for (int64_t i = start; i < end; ++i) {
            if (src[i] > 0.0) {
                ++local_cnt;
            }
        }
        thread_counts[tid] = local_cnt;
    }

    /* Compute prefix sum of counts to obtain offsets. */
    int64_t total = 0;
    for (int t = 0; t < nthreads; ++t) {
        thread_offsets[t] = total;
        total += thread_counts[t];
    }

    /* Second parallel region: write packed elements to their final positions. */
    #pragma omp parallel num_threads(nthreads) default(none) shared(src, weight, packed, LEN_1D, thread_counts, thread_offsets, nthreads)
    {
        int tid = omp_get_thread_num();
        int64_t chunk = (LEN_1D + nthreads - 1) / nthreads;
        int64_t start = tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        int64_t pos = thread_offsets[tid];
        for (int64_t i = start; i < end; ++i) {
            double s = src[i];
            if (s > 0.0) {
                packed[pos] = s * weight[i];
                ++pos;
            }
        }
    }

    if (out_count) out_count[0] = total;
    free(thread_counts);
    free(thread_offsets);
}

