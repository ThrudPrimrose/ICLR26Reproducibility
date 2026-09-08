#include <stdint.h>

/* Optimized sum reduction using OpenMP parallel for with SIMD and reduction. */
void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
    double sum = 0.0;
    #pragma omp parallel for simd reduction(+:sum)
    for (int64_t i = 0; i < LEN_1D; i++) {
        sum += a[i];
    }
    sum_out[0] = sum;
}
