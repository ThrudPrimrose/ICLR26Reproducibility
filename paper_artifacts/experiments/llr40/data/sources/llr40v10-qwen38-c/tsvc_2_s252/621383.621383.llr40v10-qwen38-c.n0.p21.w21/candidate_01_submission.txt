#include <stdint.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    /* a[i] = b[i]*c[i] + (i>0 ? b[i-1]*c[i-1] : 0.0)  -- no loop-carried dependence */
    a[0] = b[0] * c[0] + 0.0;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        double s = b[i] * c[i];
        a[i] = s + b[i - 1] * c[i - 1];
    }
}
