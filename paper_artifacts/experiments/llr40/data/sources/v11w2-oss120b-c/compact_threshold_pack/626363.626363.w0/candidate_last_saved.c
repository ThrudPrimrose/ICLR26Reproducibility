#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void compact_threshold_pack_fp64(const double *restrict src,
                                 const double *restrict weight,
                                 double *restrict packed,
                                 int64_t *restrict out_count,
                                 const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        out_count[0] = 0;
        return;
    }
    // Determine the maximum number of threads to use.
    int max_threads = omp_get_max_threads(); if (max_threads < 1) max_threads = 1; fprintf(stderr, "DEBUG start LEN=%lld max_threads=%d\n", (long long)LEN_1D, max_threads); omp_set_num_threads(1);
    // Allocate per-thread surviving element counts.
    int64_t *thread_counts = (int64_t *)calloc((size_t)max_threads, sizeof(int64_t));
    // First pass: count survivors per thread.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        fprintf(stderr, "DEBUG thread %d nt %d\n", tid, nt);
        int64_t i_start = (LEN_1D * tid) / nt;
        int64_t i_end   = (LEN_1D * (tid + 1)) / nt;
        fprintf(stderr, "DEBUG thread %d range %lld-%lld\n", tid, (long long)i_start, (long long)i_end);
        int64_t cnt = 0;
        for (int64_t i = i_start; i < i_end; ++i) {
            if (src[i] > 0.0) cnt++;
        }
        thread_counts[tid] = cnt;
        fprintf(stderr, "DEBUG thread %d cnt %lld\n", tid, (long long)cnt);

    }
    // Compute exclusive prefix sum of thread_counts to obtain output offsets.
    int64_t *thread_offsets = (int64_t *)malloc((size_t)max_threads * sizeof(int64_t));
    int64_t offset = 0;
    for (int i = 0; i < max_threads; ++i) {
        thread_offsets[i] = offset;
        offset += thread_counts[i];
    }
    fprintf(stderr, "DEBUG total offset %lld\n", (long long)offset);
    // Second pass: write packed output using the computed offsets.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        fprintf(stderr, "DEBUG thread %d nt %d\n", tid, nt);
        int64_t i_start = (LEN_1D * tid) / nt;
        int64_t i_end   = (LEN_1D * (tid + 1)) / nt;
        fprintf(stderr, "DEBUG thread %d range %lld-%lld\n", tid, (long long)i_start, (long long)i_end);
        int64_t out_idx = thread_offsets[tid];
        for (int64_t i = i_start; i < i_end; ++i) {
            if (src[i] > 0.0) {
                packed[out_idx++] = src[i] * weight[i];
            }
        }
    }
    out_count[0] = offset;
    free(thread_counts);
    free(thread_offsets);
}
