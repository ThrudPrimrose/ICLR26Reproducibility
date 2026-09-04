/*
 * compact_threshold_pack kernel implementation (C version).
 *
 * The reference algorithm packs src[i] * weight[i] into the output array `packed`
 * for each element where src[i] > 0, preserving order. The resulting count of
 * packed elements is written to out_count[0].
 *
 * This implementation uses a two‑pass parallel algorithm based on per‑thread
 * counts. The first pass counts the number of survivors per thread. After a
 * sequential prefix‑sum of those counts we obtain per‑thread offsets. The second
 * pass writes the packed values using those offsets, avoiding any atomic
 * operations and preserving the original order.
 *
 * It follows the required function signature used by the benchmark harness:
 *
 *     void compact_threshold_pack_fp64(const double *restrict src,
 *                                      const double *restrict weight,
 *                                      double *restrict packed,
 *                                      int64_t *restrict out_count,
 *                                      const int64_t LEN_1D);
 *
 * The implementation is strictly C23, uses OpenMP for parallelism, and does not
 * assume any alignment beyond the natural alignment of the input pointers.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 const double *restrict src,
                                 const double *restrict weight,
                                 const int64_t LEN_1D) {
    if (!src || !weight || !packed) {
        if (out_count) out_count[0] = 0;
        return;
    }
    if (LEN_1D <= 0) {
        if (out_count) {
            out_count[0] = 0;
        }
        return;
    }

    int max_threads = omp_get_max_threads();
    if (max_threads <= 0) max_threads = 1;
    fprintf(stderr, "compact: LEN=%lld max_threads=%d\n", (long long)LEN_1D, max_threads);
    // Allocate per‑thread survivor counts.
    int64_t *thread_counts = (int64_t*)malloc(max_threads * sizeof(int64_t));
    if (!thread_counts) {
        // Allocation failure – fall back to serial version.
        int64_t n = 0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                packed[n] = src[i] * weight[i];
                ++n;
            }
        }
        if (out_count) out_count[0] = n;
        return;
    }

    // Zero-initialize per‑thread counts.
    fprintf(stderr, "compact: allocated thread_counts of %d entries\n", max_threads);
        for (int i = 0; i < max_threads; ++i) {
            thread_counts[i] = 0;
        }
        // First pass: count survivors per thread.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t local_n = 0;
        #pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                ++local_n;
            }
        }
        thread_counts[tid] = local_n;
    }

    // Compute exclusive prefix sum of thread_counts to obtain offsets.
    int64_t total = 0;
    for (int i = 0; i < max_threads; ++i) {
        int64_t cnt = thread_counts[i];
        thread_counts[i] = total; // store offset for this thread
        total += cnt;
    }
    fprintf(stderr, "compact: total survivors=%lld\n", (long long)total);

    // Second pass: write packed elements using per‑thread offsets.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t offset = thread_counts[tid];
        int64_t local_n = 0;
        #pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                int64_t idx = offset + local_n;
                packed[idx] = src[i] * weight[i];
                ++local_n;
            }
        }
    }

    if (out_count) {
        fprintf(stderr, "compact: final out_count=%lld\n", (long long)total);
        out_count[0] = total;
    }
    free(thread_counts);
    #if 0
// Atomic parallel compaction.
    int64_t n = 0;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (src[i] > 0.0) {
            int64_t idx;
            #pragma omp atomic capture
            { idx = n; n++; }
            packed[idx] = src[i] * weight[i];
        }
    }
    out_count[0] = n;
#endif
}
