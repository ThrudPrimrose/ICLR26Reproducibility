#include <stdint.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D == 0) return;
    if (LEN_1D == 1) {
        a[0] = (b[0] + b[0] + b[0]) * 0.333;
        return;
    }
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * 0.333;
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * 0.333;
    #pragma omp parallel for simd
    for (int64_t i = 2; i < LEN_1D; i++) {
        a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333;
    }
}
