#include <stdint.h>
#include <omp.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }
    #pragma omp target
    {
        #pragma omp parallel for
        for (int64_t i = 0; i < 1024; ++i) { double x = i * 0.1; (void)x; }
    }
}
