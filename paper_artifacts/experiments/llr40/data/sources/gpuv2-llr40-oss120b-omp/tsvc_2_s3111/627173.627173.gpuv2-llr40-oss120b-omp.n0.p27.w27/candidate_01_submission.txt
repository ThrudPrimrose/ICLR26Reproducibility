#include <stdint.h>
#include <stddef.h>
#include <omp.h>

void tsvc_2_s3111_fp64(const double * restrict a, double * restrict b, const int64_t LEN_1D) {
    double sum = 0.0;
    // Offload computation to the device with parallel reduction.
    #pragma omp target teams distribute parallel for simd reduction(+:sum) schedule(static) map(to: a[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        if (ai > 0.0) {
            sum += ai;
        }
    }
    // Write result back to host memory.
    if (b != NULL) {
        b[0] = sum;
    }
}
