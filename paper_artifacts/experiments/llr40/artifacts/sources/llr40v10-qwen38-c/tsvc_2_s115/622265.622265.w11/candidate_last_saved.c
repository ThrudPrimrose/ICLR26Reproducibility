#include <stdint.h>

/* TSVC s115: for j: for i>j: a[i] -= aa[j,i] * a[j].
 * j-chain carries dependence (a[j] updated by all earlier rows),
 * inner i-loop independent. v1: per-row parallel. */
void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n <= 1024) {
    for (int64_t j = 0; j < n; j++) {
      const double aj = a[j];
      const double *row = aa + j * n;
      for (int64_t i = j + 1; i < n; i++)
        a[i] -= row[i] * aj;
    }
    return;
  }
  #pragma omp parallel
  {
    for (int64_t j = 0; j < n; j++) {
      const double aj = a[j];
      const double *row = aa + j * n;
      #pragma omp for
      for (int64_t i = j + 1; i < n; i++)
        a[i] -= row[i] * aj;
    }
  }
}
