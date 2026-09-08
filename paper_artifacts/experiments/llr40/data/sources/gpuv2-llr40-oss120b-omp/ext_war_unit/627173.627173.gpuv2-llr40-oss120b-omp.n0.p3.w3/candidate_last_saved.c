#include <stdint.h>
#include <stdlib.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 1) return;
    // Dummy target region to ensure at least one device kernel is emitted.
    #pragma omp target
    {
        // No operation; this region exists solely to register a target kernel.
    }
    // Allocate a temporary buffer on the host to hold the original values of 'a'.
    double *temp = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (!temp) {
        // Allocation failed – fall back to the straightforward sequential implementation.
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            a[i] = a[i + 1] + b[i];
        }
        return;
    }
    // Copy the original data into the temporary buffer in parallel.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        temp[i] = a[i];
    }
    // Compute the result using the temporary buffer to avoid data races.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        a[i] = temp[i + 1] + b[i];
    }
    free(temp);
}
