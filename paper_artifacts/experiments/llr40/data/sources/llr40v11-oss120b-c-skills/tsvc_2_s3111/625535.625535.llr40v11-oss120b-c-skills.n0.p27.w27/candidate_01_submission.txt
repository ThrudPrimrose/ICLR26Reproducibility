#include <stdint.h>
#include <omp.h>

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    double sum = 0.0;
    #pragma omp parallel for simd reduction(+:sum) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        if (ai > 0.0) {
            sum += ai;
        }
    }
    b[0] = sum;
}
