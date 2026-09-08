#include <stdint.h>
#include <omp.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    // If no OpenMP target device is present, fall back to host parallel execution.
    if (omp_get_num_devices() == 0) {
        #pragma omp parallel for simd schedule(static) default(none) shared(a,b,c,LEN_1D)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] = a[i] * b[i] * c[i];
        }
        return;
    }

    // Offload computation to the GPU device.
    #pragma omp target teams distribute parallel for simd map(to: b[0:LEN_1D], c[0:LEN_1D]) map(tofrom: a[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }
}

