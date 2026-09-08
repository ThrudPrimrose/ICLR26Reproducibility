#include <stdint.h>
#include <omp.h>

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;

    int64_t n4 = (LEN_1D / 4) * 4;

    #pragma omp parallel for simd reduction(+:s0,s1,s2,s3) schedule(static)
    for (int64_t i = 0; i < n4; i += 4) {
        s0 += a[i + 0];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }

    double sum = s0 + s1 + s2 + s3;

    #pragma omp simd reduction(+:sum)
    for (int64_t i = n4; i < LEN_1D; i++) {
        sum += a[i];
    }

    sum_out[0] = sum;
}
