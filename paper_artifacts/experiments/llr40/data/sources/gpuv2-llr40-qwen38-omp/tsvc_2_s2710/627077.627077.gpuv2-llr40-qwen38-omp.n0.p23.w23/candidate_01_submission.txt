#pragma STDC FP_CONTRACT OFF
#include <stdint.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c, const double *restrict d,
                       const double *restrict e, const double *restrict x, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  #pragma omp target teams distribute parallel for \
      map(to: d[0:n], e[0:n], x[0:1]) map(tofrom: a[0:n], b[0:n], c[0:n])
  for (int64_t i = 0; i < n; ++i) {
    if (a[i] > b[i]) {
      a[i] += b[i] * d[i];
      if (n > 10) c[i] += d[i] * d[i];
      else c[i] = d[i] * e[i] + 1.0;
    } else {
      b[i] = a[i] + e[i] * e[i];
      if (x[0] > 0.0) c[i] = a[i] + d[i] * d[i];
      else c[i] += e[i] * e[i];
    }
  }
}
