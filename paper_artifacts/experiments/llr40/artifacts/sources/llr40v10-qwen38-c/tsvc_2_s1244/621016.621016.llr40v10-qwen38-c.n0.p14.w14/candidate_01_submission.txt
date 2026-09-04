#include <stdint.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
  const int64_t n = LEN_1D - 1;
  if (n <= 0) return;
  /*
   * Reference: for i in [0, LEN-2]: a[i] = e(i); d[i] = a[i] + a[i+1];
   * Forward sequential execution reads a[i+1] BEFORE it is overwritten at
   * iteration i+1, so semantically:
   *   d[i] = e(i) + a_old[i+1]   (a_old = input array)
   *   a[i] = e(i)  for i < LEN-1; a[LEN-1] untouched.
   * Both are elementwise independent -> two parallel loops, d first so it
   * sees the original a.
   */
  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < n; i++) {
      d[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i] + a[i + 1];
    }
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < n; i++) {
      a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
    }
  }
}
