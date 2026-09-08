#include <stdint.h>

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
    double sum = 0.0;
    // Parallel reduction using OpenMP. The reduction clause ensures thread-local accumulation.
    #pragma omp parallel for reduction(+:sum)
    for (int64_t i = 0; i < LEN_1D; i++) {
        sum += a[i];
    }
    sum_out[0] = sum;
}
