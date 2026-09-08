/*
 * Optimized implementation of scatter_accum_dup kernel.
 * Performs indexed accumulation bins[ip[i]] += src[i] for i = 0 .. LEN_1D-1.
 * Handles the case where ip is a permutation (no duplicate indices) by using a race-free
 * parallel loop without atomics, and falls back to an atomic loop when duplicates may exist.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

void scatter_accum_dup_fp64(const double *restrict src,
                            const int32_t *restrict ip,
                            double *restrict bins,
                            const int64_t LEN_1D) {
    bool duplicate = false;
    if (LEN_1D > 0) {
        uint8_t *visited = (uint8_t *)calloc((size_t)LEN_1D, sizeof(uint8_t));
        if (visited == NULL) {
            duplicate = true;
        } else {
            for (int64_t i = 0; i < LEN_1D; ++i) {
                int32_t idx = ip[i];
                if (visited[(size_t)idx]) {
                    duplicate = true;
                    break;
                }
                visited[(size_t)idx] = 1;
            }
            free(visited);
        }
    }
    if (!duplicate) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            int32_t idx = ip[i];
            bins[idx] += src[i];
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            int32_t idx = ip[i];
            #pragma omp atomic update
            bins[idx] += src[i];
        }
    }
}
