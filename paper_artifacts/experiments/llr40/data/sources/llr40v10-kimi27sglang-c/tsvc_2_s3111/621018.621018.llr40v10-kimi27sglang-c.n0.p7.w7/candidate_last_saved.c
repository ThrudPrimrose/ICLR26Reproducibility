#include <stdint.h>

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double v = a[i];
        sum += (v > 0.0) ? v : 0.0;
    }
    b[0] = sum;
}
