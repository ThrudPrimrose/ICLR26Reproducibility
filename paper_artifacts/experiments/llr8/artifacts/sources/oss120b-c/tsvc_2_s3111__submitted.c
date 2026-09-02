#include <stddef.h>
#include <stdint.h>

void tsvc_2_s3111_fp64(double *restrict a, double *restrict b, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    double sum_val = 0.0;
    #pragma omp simd reduction(+:sum_val)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        if (ai > 0.0) {
            sum_val += ai;
        }
    }
    b[0] = sum_val;
    (void)workspace;
    (void)workspace_bytes;
}
