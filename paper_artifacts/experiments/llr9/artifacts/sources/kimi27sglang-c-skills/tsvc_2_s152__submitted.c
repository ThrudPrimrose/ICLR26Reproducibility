#include <stdint.h>
#include <omp.h>

void tsvc_2_s152_fp64(double* restrict a, double* restrict b, double* restrict c,
                      double* restrict d, double* restrict e, int64_t LEN_1D,
                      uint8_t* restrict workspace, int64_t workspace_bytes)
{
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double bi = d[i] * e[i];
        b[i] = bi;
        a[i] = a[i] + bi * c[i];
    }
}
