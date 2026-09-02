#include <stddef.h>
#include <stdint.h>

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, double *restrict c, double *restrict d, double *restrict e, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_size) {
    #pragma omp parallel for
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double t = d[i] * e[i];
        b[i] = t;
        a[i] = a[i] + t * c[i];
    }
}
