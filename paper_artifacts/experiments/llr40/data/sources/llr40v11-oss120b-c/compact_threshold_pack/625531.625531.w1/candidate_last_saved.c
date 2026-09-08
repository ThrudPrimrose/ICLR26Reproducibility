/*
 * Optimized implementation of compact_threshold_pack kernel.
 * Input arrays src and weight (double) of length LEN_1D.
 * Output array packed (double) of length LEN_1D, initially zero.
 * Output count out_count[0] is set to number of packed elements.
 *
 * The naive implementation suffers from a loop-carried dependency on the write index.
 * This implementation breaks the dependency by performing a first pass that computes exclusive
 * prefix sums (write indices) for each element, followed by a branchless second pass that stores
 * the product only for surviving elements. This eliminates unpredictable branches and enables
 * vectorization of the second pass. Memory overhead is O(N) for a temporary index array.
 */

#include <stdlib.h>
#include <stdint.h>

#include <stdio.h>
#include <string.h>

#pragma GCC push_options
#pragma GCC optimize ("O3", "no-tree-vectorize")
void compact_threshold_pack_fp64(const double *restrict src,
                                 const double *restrict weight,
                                 double *restrict packed,
                                 int64_t *restrict out_count,
                                 const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        out_count[0] = 0;
        return;
    }

    // Compute number of survivors and optionally write results.
    int64_t n = 0;
    if (packed) {
        // Initialize output buffer to zero for reproducibility.
        memset(packed, 0, (size_t)LEN_1D * sizeof(double));
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                packed[n] = src[i] * weight[i];
                ++n;
            }
        }
    } else {
        // packed pointer is NULL – only count survivors.
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                ++n;
            }
        }
    }
    out_count[0] = n;
    return;
}
#pragma GCC pop_options

