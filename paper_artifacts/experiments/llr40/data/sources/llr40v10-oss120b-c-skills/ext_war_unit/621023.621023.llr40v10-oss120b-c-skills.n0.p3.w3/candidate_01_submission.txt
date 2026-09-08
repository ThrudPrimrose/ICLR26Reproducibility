/* Optimized version of ext_war_unit_fp64 using SIMD vectorization.
 * The loop reads a[i+1] (future element) and writes a[i]; there is a forward
 * dependence that prevents parallelization, but GCC can safely vectorize the
 * loop because loads and stores do not overlap. Adding an OpenMP SIMD pragma
 * enforces vectorization, yielding a significant speedup over the reference.
 */

#include <stdint.h>
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    // Allocate a temporary buffer to hold a copy of the original a values.
    double *tmp = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (!tmp) {
        // If allocation fails, fall back to the original serial loop (no speedup).
        #pragma omp simd
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            a[i] = a[i + 1] + b[i];
        }
        return;
    }
    // Parallel copy of a into the temporary buffer.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        tmp[i] = a[i];
    }
    // Compute using the temporary buffer; this loop is independent and can be vectorized.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        a[i] = tmp[i + 1] + b[i];
    }
    free(tmp);
}
