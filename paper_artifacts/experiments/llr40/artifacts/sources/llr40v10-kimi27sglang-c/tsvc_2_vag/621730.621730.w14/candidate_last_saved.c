#include <stdint.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    int64_t i = 0;
    int64_t imax = LEN_1D >= 4 ? LEN_1D - 3 : 0;
#pragma omp parallel for schedule(static)
    for (i = 0; i < imax; i += 4) {
        double r0 = b[ip[i]];
        double r1 = b[ip[i + 1]];
        double r2 = b[ip[i + 2]];
        double r3 = b[ip[i + 3]];
        a[i] = r0;
        a[i + 1] = r1;
        a[i + 2] = r2;
        a[i + 3] = r3;
    }
    for (; i < LEN_1D; ++i) {
        a[i] = b[ip[i]];
    }
}
