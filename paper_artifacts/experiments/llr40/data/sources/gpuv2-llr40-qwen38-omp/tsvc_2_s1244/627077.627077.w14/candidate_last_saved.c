#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
  const int64_t N = LEN_1D;
  const int64_t n = N - 1;
  if (n <= 0) return;

  #pragma omp target map(tofrom: a[0:N]) map(to: b[0:N]) map(to: c[0:N]) map(from: d[0:n])
  {
    /* d[i] = (b[i] + c[i]*c[i] + b[i]*b[i] + c[i]) + a[i+1]; a is untouched here,
       so a[i+1] is still the original value, matching the sequential reference. */
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
      const double x = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
      d[i] = x + a[i + 1];
    }

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
      a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
    }
  }
}
