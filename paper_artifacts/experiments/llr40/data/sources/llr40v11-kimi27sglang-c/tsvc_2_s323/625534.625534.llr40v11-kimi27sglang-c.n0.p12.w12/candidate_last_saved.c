#include <stdint.h>
#include <math.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {

  for (int64_t i = 1; i < LEN_1D; ++i) {
    double ai = fma(c[i], d[i], b[i - 1]);
    double bi = fma(c[i], e[i], ai);
    a[i] = ai;
    b[i] = bi;
  }
}
