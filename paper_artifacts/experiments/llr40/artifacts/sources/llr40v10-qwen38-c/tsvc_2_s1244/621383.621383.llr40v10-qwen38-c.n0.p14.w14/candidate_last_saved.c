#include <stdint.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
  const int64_t n = LEN_1D - 1;
  if (n <= 0) return;
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) d[i] = a[i + 1];
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) {
    const double f = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
    d[i] += f;
    a[i] = f;
  }
}
