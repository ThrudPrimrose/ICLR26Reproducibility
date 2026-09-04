#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>

/*
 * Scatter accumulate with possible duplicate indices.
 *
 * bins   : array of double values to be updated (in-place).
 * src    : source values, one per index.
 * ip     : integer indices (int32_t) into bins.
 * LEN_1D : length of the arrays.
 *
 * The index array may contain duplicates. When it is a permutation (i.e., all indices are
 * unique), we can safely parallelise the loop without atomics. Otherwise we fall back to an
 * atomic addition to preserve correctness.
 */
void scatter_accum_dup_fp64(double *restrict bins,
                            const int32_t *restrict ip,
                            const double *restrict src,
                            const int64_t LEN_1D) {
    if (LEN_1D <= 0)
        return;

    // Quick test for permutation (no duplicate indices).
    // Allocate a byte array to mark visited indices.
    // Using calloc zero-initialises the array.
    unsigned char *visited = (unsigned char *)calloc((size_t)LEN_1D, sizeof(unsigned char));
    bool is_perm = true;
    if (visited != NULL) {
        for (int64_t i = 0; i < LEN_1D; ++i) {
            int32_t idx = ip[i];
            // The generator guarantees 0 <= idx < LEN_1D, but we guard against out‑of‑range.
            if (idx < 0 || idx >= LEN_1D) {
                // Out‑of‑range indices are treated as duplicates to force the atomic path.
                is_perm = false;
                break;
            }
            if (visited[(size_t)idx]) {
                is_perm = false;
                break;
            }
            visited[(size_t)idx] = 1;
        }
        free(visited);
    } else {
        // Allocation failed – fall back to the safe (atomic) path.
        is_perm = false;
    }

    if (is_perm) {
        // No duplicates: we can safely parallelise without atomics.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            bins[ip[i]] += src[i];
        }
    } else {
        // Duplicates exist: use atomic updates.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            #pragma omp atomic
            bins[ip[i]] += src[i];
        }
    }
}

