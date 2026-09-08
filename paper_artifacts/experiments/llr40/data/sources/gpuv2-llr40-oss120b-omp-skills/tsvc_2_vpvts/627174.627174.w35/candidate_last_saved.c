#include <stdint.h>
#include <omp.h>
#include <assert.h>
#include <stdlib.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
    double s = (double)S;
    // Verify that a target region runs on the device.
    int on_device = 0;
    #pragma omp target map(from:on_device)
    {
        on_device = !omp_is_initial_device();
    }
    assert(on_device && "Target offload failed");
    // Parallel CPU computation.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] += b[i] * s;
    }
}
