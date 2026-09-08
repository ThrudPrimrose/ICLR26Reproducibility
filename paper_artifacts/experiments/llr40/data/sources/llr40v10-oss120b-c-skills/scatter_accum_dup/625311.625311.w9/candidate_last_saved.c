/* scatter_accum_dup kernel implementation (C version).
 * Accumulates src values into bins according to index array ip.
 * Handles duplicate indices safely using atomic updates.
 * If ip contains no duplicates (a permutation), a fast parallel loop without atomics is used.
 *
 * Signature follows the benchmark ABI: all pointers are restrict-qualified where appropriate.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

void scatter_accum_dup_fp64(double *restrict bins, const double *restrict src,
                            const int32_t *restrict ip, const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        return;
    }

    // Detect duplicates in ip: allocate a visited bitmap of LEN_1D bytes.
    bool has_dup = false;
    // Use calloc to get zero-initialized memory.
    unsigned char *visited = (unsigned char *)calloc((size_t)LEN_1D, sizeof(unsigned char));
    if (visited != NULL) {
        for (int64_t i = 0; i < LEN_1D; ++i) {
            int32_t idx = ip[i];
            // Guard against out-of-bounds indices (should not happen per spec).
            if ((int64_t)idx < 0 || idx >= LEN_1D) {
                // Treat as duplicate to fall back to safe atomic path.
                has_dup = true;
                break;
            }
            if (visited[(size_t)idx]) {
                has_dup = true;
                break;
            }
            visited[(size_t)idx] = 1;
        }
        free(visited);
    } else {
        // Allocation failure – be safe and use atomic updates.
        has_dup = true;
    }

    if (has_dup) {
        // Use atomic updates to avoid lost increments when indices collide.
        #pragma omp parallel for schedule(static) default(none) \
            shared(bins, src, ip, LEN_1D)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            int32_t idx = ip[i];
            #pragma omp atomic
            bins[idx] += src[i];
        }
    } else {
        // No duplicates – a plain parallel loop suffices.
        #pragma omp parallel for schedule(static) default(none) \
            shared(bins, src, ip, LEN_1D)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            bins[ip[i]] += src[i];
        }
    }
}

