#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void tsvc_2_vtvtv_fp64(
    double *restrict a,
    const double *restrict b,
    const double *restrict c,
    const int64_t LEN_1D,
    uint8_t *restrict workspace,
    const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    if (LEN_1D < (1 << 18)) {
        for (int64_t i = 0; i < LEN_1D; ++i)
            a[i] = a[i] * b[i] * c[i];
    } else {
        #pragma omp parallel for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i)
            a[i] = a[i] * b[i] * c[i];
    }
}
