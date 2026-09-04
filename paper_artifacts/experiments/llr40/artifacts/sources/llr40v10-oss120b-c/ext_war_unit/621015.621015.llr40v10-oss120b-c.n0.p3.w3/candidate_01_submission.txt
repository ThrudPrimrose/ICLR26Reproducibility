#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    // No work for zero or one element
    if (LEN_1D <= 1) {
        return;
    }

    // Allocate an aligned temporary buffer to hold computed results.
    // Alignment of 64 bytes matches typical cache line / SIMD width.
    double *tmp = (double*)aligned_alloc(64, (size_t)LEN_1D * sizeof(double));
    if (!tmp) {
        // Fallback to sequential algorithm if allocation fails.
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            a[i] = a[i + 1] + b[i];
        }
        return;
    }

    // Parallel region: compute and copy back in two independent loops.
    #pragma omp parallel
    {
        // Compute each output element independently.
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            tmp[i] = a[i + 1] + b[i];
        }
        // Copy results back to the output array.
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            a[i] = tmp[i];
        }
    }

    free(tmp);
}
