/*
 * Simple serial implementation of scatter_accum_dup.
 * Performs indexed accumulation: bins[ip[i]] += src[i]
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void scatter_accum_dup_fp64(double *restrict bins,
                           const double *restrict src,
                           const int32_t *restrict ip,
                           const int64_t LEN_1D) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        int64_t raw = (int64_t)ip[i];
        int64_t idx = raw % LEN_1D;
        if (idx < 0) idx += LEN_1D;
        #pragma omp atomic
        bins[idx] += src[i];
    }
}
