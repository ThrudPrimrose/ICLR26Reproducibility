#include <stdint.h>
#include <omp.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, double *restrict c, double *restrict d, double *restrict e, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    double sum_val = 0.0;
    #pragma omp parallel for simd reduction(+:sum_val) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double c_val = c[i];
        double a_val = c_val + d[i];
        a[i] = a_val;
        double b_val = c_val + e[i];
        b[i] = b_val;
        sum_val += a_val + b_val;
    }
    b[0] = sum_val;
}
