// Compact Threshold Pack kernel implementation in C++.
//
// This kernel mirrors the NumPy reference implementation located at
// /shared/tasks/compact_threshold_pack/compact_threshold_pack_numpy.py.
//
// The kernel packs the product src[i] * weight[i] for every element where
// src[i] > 0.0, preserving the original order.  The number of packed elements
// is written to out_count[0].  The input arrays are double‑precision; the
// harness calls the function with the "_fp64" suffix.
//
// The implementation uses a two‑pass parallel algorithm based on OpenMP:
//   1. Count survivors per thread.
//   2. Compute a prefix sum of those counts to obtain per‑thread output
//      offsets.
//   3. Write the packed values in a second parallel loop using the offsets.
//
// This approach avoids atomics, runs in O(N) work, and scales with the number
// of OpenMP threads.  It respects the required order because each thread
// processes a contiguous block of the input (static scheduling) and the final
// offsets reflect the total number of survivors in all preceding threads.
//
// The function signature follows the convention used by other kernels in the
// benchmark (e.g., argmax_with_index_fp64).  The "extern \"C\"" linkage avoids
// name mangling so the harness can locate the symbol.

#include <cstdint>
#include <vector>
#include <omp.h>
#include <cstdio>

extern "C" void compact_threshold_pack_fp64(const double* __restrict__ src,
                                            const double* __restrict__ weight,
                                            double* __restrict__ packed,
                                            int64_t* __restrict__ out_count,
                                            const int64_t LEN_1D) {
    // First pass: count positive entries per thread.
    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;
    omp_set_dynamic(0);
    std::vector<int64_t> thread_counts(num_threads, 0);
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int64_t local_count = 0;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                ++local_count;
            }
        }
        thread_counts[tid] = local_count;
    }

    // Compute exclusive prefix sum of the per-thread counts to get offsets.
    std::vector<int64_t> thread_offsets(num_threads, 0);
    int64_t offset = 0;
    for (int t = 0; t < num_threads; ++t) {
        thread_offsets[t] = offset;
        offset += thread_counts[t];
    }
    int64_t total = offset;
    out_count[0] = total;
    // Debug output
    printf("out_count=%lld\n", (long long)total);

    // Second pass: write packed values using the computed offsets.
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int64_t pos = thread_offsets[tid];
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                packed[pos++] = src[i] * weight[i];
            }
        }
    }
}
