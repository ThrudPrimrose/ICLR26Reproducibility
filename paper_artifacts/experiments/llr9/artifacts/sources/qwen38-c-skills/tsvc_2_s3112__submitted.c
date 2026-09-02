#include <stdint.h>
#include <omp.h>

void tsvc_2_s3112_fp64(double *a, double *b, int64_t LEN_1D,
                       uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    double sum = 0.0;
    for (int64_t i = 0; i < LEN_1D; i++) {
        sum = sum + a[i];
        b[i] = sum;
    }
}
