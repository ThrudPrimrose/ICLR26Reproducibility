#include <stdint.h>
#include <omp.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    if (LEN_1D <= 1) return;

    double s = b[0];
    double prev, cd, ce;
    #pragma omp parallel for reduction(inscan,+:s) private(prev, cd, ce)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        prev = s;
        cd = c[i] * d[i];
        ce = c[i] * e[i];
        s = s + cd + ce;
        #pragma omp scan inclusive(s)
        b[i] = s;
        a[i] = prev + cd;
    }
}
