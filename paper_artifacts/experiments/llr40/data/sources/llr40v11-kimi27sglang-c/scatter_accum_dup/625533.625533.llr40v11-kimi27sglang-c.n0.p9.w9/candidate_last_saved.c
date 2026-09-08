#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void scatter_accum_dup_fp64(double *restrict bins, const int32_t *restrict ip,
                              const double *restrict src, int64_t LEN_1D,
                              uint8_t *restrict workspace, int64_t workspace_bytes) {
    if (LEN_1D <= 0) return;

    /* Fast path: ip is a permutation -> no conflicts.
       Re-use the provided scratch as a byte mask to avoid a malloc. */
    bool is_perm = true;
    char *seen = NULL;
    bool seen_from_workspace = (workspace_bytes >= LEN_1D);
    if (seen_from_workspace) {
        seen = (char *)workspace;
        memset(seen, 0, (size_t)LEN_1D);
    } else {
        seen = (char *)calloc((size_t)LEN_1D, sizeof(char));
        if (seen == NULL) {
            /* Allocation failed: fall back to atomics without checking. */
            is_perm = false;
        }
    }

    if (is_perm) {
        for (int64_t i = 0; i < LEN_1D; ++i) {
            int64_t idx = (int64_t)ip[i];
            if (idx < 0 || idx >= LEN_1D || seen[idx]) {
                is_perm = false;
                break;
            }
            seen[idx] = 1;
        }
    }

    if (!seen_from_workspace && seen != NULL) free(seen);

    if (is_perm) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            bins[ip[i]] += src[i];
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            #pragma omp atomic
            bins[ip[i]] += src[i];
        }
    }
}
