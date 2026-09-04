#include <stdint.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    const double factor = 0.333;
    if (LEN_1D <= 0) {
        return;
    }
    if (LEN_1D == 1) {
        a[0] = (b[0] * 3.0) * factor;
        return;
    }
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * factor;
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * factor;
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 2; i < LEN_1D; ++i) {
        a[i] = (b[i] + b[i-1] + b[i-2]) * factor;
    }
}
