#include <stdint.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {
    int64_t n = LEN_2D * LEN_2D;
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < n; ++idx) {
        aa[idx] = aa[idx] + bb[idx] * cc[idx];
    }
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] = b[i] + c[i] * d[i];
    }
}
