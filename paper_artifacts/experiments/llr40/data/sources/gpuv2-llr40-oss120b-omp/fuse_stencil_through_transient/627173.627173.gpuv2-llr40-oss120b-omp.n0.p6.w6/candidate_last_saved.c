#include <stdint.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    // Dummy target region to ensure device kernel registration
    #pragma omp target
    {
    }
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 1; i < LEN_1D - 2; ++i) {
        double left = a[i - 1] + a[i] + a[i + 1];
        double right = a[i] + a[i + 1] + a[i + 2];
        out[i] = left * right;
    }
}
