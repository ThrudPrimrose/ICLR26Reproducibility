#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n <= 0) return;
  const double x0 = x[0];

  if (n > 10) {
    if (x0 > 0.0) {
      #pragma omp target map(tofrom: a[0:n], b[0:n], c[0:n]) map(to: d[0:n], e[0:n]) teams distribute parallel for simd
      for (int64_t i = 0; i < n; ++i) {
        if (a[i] > b[i]) {
          a[i] += b[i] * d[i];
          c[i] += d[i] * d[i];
        } else {
          b[i] = a[i] + e[i] * e[i];
          c[i] = a[i] + d[i] * d[i];
        }
      }
    } else {
      #pragma omp target map(tofrom: a[0:n], b[0:n], c[0:n]) map(to: d[0:n], e[0:n]) teams distribute parallel for simd
      for (int64_t i = 0; i < n; ++i) {
        if (a[i] > b[i]) {
          a[i] += b[i] * d[i];
          c[i] += d[i] * d[i];
        } else {
          b[i] = a[i] + e[i] * e[i];
          c[i] += e[i] * e[i];
        }
      }
    }
  } else {
    if (x0 > 0.0) {
      #pragma omp target map(tofrom: a[0:n], b[0:n], c[0:n]) map(to: d[0:n], e[0:n])
      for (int64_t i = 0; i < n; ++i) {
        if (a[i] > b[i]) {
          a[i] += b[i] * d[i];
          c[i] = d[i] * e[i] + 1.0;
        } else {
          b[i] = a[i] + e[i] * e[i];
          c[i] = a[i] + d[i] * d[i];
        }
      }
    } else {
      #pragma omp target map(tofrom: a[0:n], b[0:n], c[0:n]) map(to: d[0:n], e[0:n])
      for (int64_t i = 0; i < n; ++i) {
        if (a[i] > b[i]) {
          a[i] += b[i] * d[i];
          c[i] = d[i] * e[i] + 1.0;
        } else {
          b[i] = a[i] + e[i] * e[i];
          c[i] += e[i] * e[i];
        }
      }
    }
  }
}
