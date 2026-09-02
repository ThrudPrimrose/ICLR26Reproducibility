#include <stdint.h>
#include <omp.h>

void tsvc_2_s311_fp64(double * restrict a, double * restrict sum_out, int64_t LEN_1D, uint8_t * restrict workspace, int64_t workspace_bytes) {
    (void)workspace; // unused
    (void)workspace_bytes;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        sum += a[i];
    }
    sum_out[0] = sum;
}
