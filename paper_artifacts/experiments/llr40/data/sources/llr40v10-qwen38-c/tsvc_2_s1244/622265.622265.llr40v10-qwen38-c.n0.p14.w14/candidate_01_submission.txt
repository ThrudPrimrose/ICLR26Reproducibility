#include <stdint.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
  const int64_t n = LEN_1D - 1;
  if (n <= 0) return;
  a[0] = b[0] + c[0] * c[0] + b[0] * b[0] + c[0];
#pragma omp parallel for schedule(static)
  for (int64_t j = 1; j < n; j++) {
    const double oldj = a[j];
    a[j] = b[j] + c[j] * c[j] + b[j] * b[j] + c[j];
    d[j - 1] = b[j - 1] + c[j - 1] * c[j - 1] + b[j - 1] * b[j - 1] + c[j - 1] + oldj;
  }
  d[n - 1] = a[n - 1] + a[n];
}
