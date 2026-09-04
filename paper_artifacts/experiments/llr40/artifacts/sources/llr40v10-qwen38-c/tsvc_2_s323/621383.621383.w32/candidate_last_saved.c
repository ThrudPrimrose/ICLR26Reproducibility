#include <stdint.h>
#include <stdio.h>
void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  for (int64_t i = 1; i < LEN_1D; ++i) {
    a[i] = b[i - 1] + c[i] * d[i];
    b[i] = a[i] + c[i] * e[i];
  }
  printf("N=%lld c0=%.3g c1=%.3g d1=%.3g e1=%.3g b0=%.3g\n", (long long)LEN_1D, c[0], c[1], d[1], e[1], b[0]);
}
