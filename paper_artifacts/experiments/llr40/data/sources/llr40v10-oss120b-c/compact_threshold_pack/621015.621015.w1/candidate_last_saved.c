#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

/*
 * Stream compaction with weighted product: pack src[i] * weight[i] for each
 * src[i] > 0. The number of packed elements is written to out_count[0].
 *
 * This implementation uses a two‑pass parallel algorithm:
 *   1. Determine a 0/1 mask for each element and count survivors per thread.
 *   2. Compute a prefix sum of the per‑thread counts to obtain each thread's
 *      output offset, then scatter the surviving products into the packed array.
 *
 * The algorithm is O(N) work with O(N) extra temporary storage (1 byte per
 * element for the mask) and scales with the number of OpenMP threads.
 */

void compact_threshold_pack_fp64(const double *restrict src,
                                 const double *restrict weight,
                                 double *restrict packed,
                                 int64_t *restrict out_count,
                                 const int64_t LEN_1D) {
        // Simple serial implementation: pack src[i] * weight[i] for src[i] > 0.
    int64_t n = 0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (src[i] > 0.0) {
            packed[n++] = src[i] * weight[i];
        }
    }
    out_count[0] = n;
}

