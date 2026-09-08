#include <stdint.h>
#include <omp.h>

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    // Check we are actually offloading
    int on_device = 0;
    #pragma omp target map(from:on_device)
    {
        on_device = !omp_is_initial_device();
    }
    // Offload data and compute
    #pragma omp target data map(to: b[0:LEN_1D], ip[0:LEN_1D]) map(tofrom: a[0:LEN_1D])
    {
        #pragma omp target teams distribute parallel for simd
        for (int64_t i = 0; i < LEN_1D; ++i) {
            int64_t idx = (int64_t)ip[i];
            a[i] += b[idx] * 2.0;
        }
    }
}
