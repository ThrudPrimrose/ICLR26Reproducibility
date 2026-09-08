#include <stdint.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    // Compute out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])
    // i runs from 1 to LEN_1D-3 inclusive.
    // Use OpenMP parallel for with SIMD.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 1; i < LEN_1D - 2; ++i) {
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
}
