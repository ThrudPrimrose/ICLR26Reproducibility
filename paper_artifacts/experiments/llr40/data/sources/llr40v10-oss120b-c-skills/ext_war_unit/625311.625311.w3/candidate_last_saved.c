#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    // Edge case: if length less than 2, nothing to do (a unchanged)
    if (LEN_1D <= 1) return;
    // Allocate a temporary copy of a to read original values
    double *tmp = (double*) aligned_alloc(64, (size_t)LEN_1D * sizeof(double));
    if (!tmp) {
        // fallback to malloc if aligned_alloc fails
        tmp = (double*) malloc((size_t)LEN_1D * sizeof(double));
        if (!tmp) return; // allocation failure, do nothing
    }
    // Copy original a to tmp in parallel
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        tmp[i] = a[i];
    }
    // Compute new values for a[0..LEN_1D-2]
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        a[i] = tmp[i + 1] + b[i];
    }
    // a[LEN_1D-1] unchanged (already contains original value)
    // Free temporary buffer
    free(tmp);
}
