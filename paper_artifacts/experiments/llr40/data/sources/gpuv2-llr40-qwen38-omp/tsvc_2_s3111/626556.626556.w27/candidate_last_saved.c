#include <stdint.h>

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    double sum = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:sum) map(to: a[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > 0.0) sum += a[i];
    }
    b[0] = sum;
}
