#include <stdint.h>
#include <omp.h>

void tsvc_2_vpvts_fp64(double * restrict a, const double * restrict b, int64_t LEN_1D, int64_t S, uint8_t * restrict workspace, int64_t workspace_bytes) {
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = a[i] + b[i] * (double)S;
    }
}
