#include <stdint.h>
#include <cblas.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b,
                       const int64_t LEN_1D, const int64_t S) {
    if (LEN_1D <= 0) return;
    cblas_daxpy((int)LEN_1D, (double)S, b, 1, a, 1);
}
