#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s3112_fp64(double *restrict a, double *restrict b, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    double sum = 0.0;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        sum += a[i];
        b[i] = sum;
    }
}
