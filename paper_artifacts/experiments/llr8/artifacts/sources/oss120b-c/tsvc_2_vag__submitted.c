#include <stddef.h>
#include <stdint.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b,
                      const int32_t *restrict ip, int64_t LEN_1D,
                      uint8_t *restrict workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = b[ip[i]];
    }
}
