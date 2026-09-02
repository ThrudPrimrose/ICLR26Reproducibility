#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double * restrict aa, double * restrict bb, int64_t LEN_2D, uint8_t * restrict workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    #pragma omp parallel
    for (int64_t j = 1; j < LEN_2D; j++) {
        double * restrict a_row = aa + j * LEN_2D;
        const double * restrict a_prev = aa + (j - 1) * LEN_2D;
        const double * restrict b_row = bb + j * LEN_2D;
        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < LEN_2D; i++) {
            a_row[i] = a_prev[i] + b_row[i];
        }
    }
}
