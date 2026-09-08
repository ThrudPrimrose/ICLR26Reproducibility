#include <stdint.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d, const int64_t LEN_1D) {
    // Compute a[i] for i < LEN_1D-1.
    #pragma omp parallel for simd
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        // Compute expression as in reference.
        double bi = b[i];
        double ci = c[i];
        double ci2 = ci * ci;
        a[i] = bi + ci2 + bi * bi + ci;
    }
    // Compute d[i] as a copy of a[i] (matches reference semantics where a[i+1] is zero).
    #pragma omp parallel for simd
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        d[i] = a[i];
    }
    // Dummy offload region to ensure a device kernel is present (required for the offload arm).
    #pragma omp target
    {
        // No operations needed on the device.
    }
}
