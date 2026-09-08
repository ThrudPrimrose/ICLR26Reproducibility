#include <stdint.h>
#include <omp.h>

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
    double s = 0.0;
    if (LEN_1D > (1 << 18)) {
        #pragma omp target map(to: a[0:LEN_1D])
        #pragma omp teams distribute parallel for simd reduction(+:s)
        for (int64_t i = 0; i < LEN_1D; i++) s += a[i];
    } else {
        #pragma omp simd reduction(+:s)
        for (int64_t i = 0; i < LEN_1D; i++) s += a[i];
    }
    sum_out[0] = s;
}
