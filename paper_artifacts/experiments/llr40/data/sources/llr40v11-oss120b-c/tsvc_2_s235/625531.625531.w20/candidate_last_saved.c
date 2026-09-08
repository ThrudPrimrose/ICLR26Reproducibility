#include <stdint.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        double a_i = a[i] + b[i] * c[i];
        a[i] = a_i;
        double accum = aa[i];
        for (int64_t j = 1; j < LEN_2D; ++j) {
            int64_t idx = j * LEN_2D + i;
            accum += bb[idx] * a_i;
            aa[idx] = accum;
        }
    }
}
