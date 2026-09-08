#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
  const int x0pos = x[0] > 0.0;
  const int64_t n32 = (LEN_1D / 32) * 32;
  if (LEN_1D > 10) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n32; i += 32) {
      for (int64_t k = 0; k < 32; ++k) {
        const int64_t idx = i + k;
        if (a[idx] > b[idx]) {
          a[idx] += b[idx] * d[idx];
          c[idx] += d[idx] * d[idx];
        } else {
          b[idx] = a[idx] + e[idx] * e[idx];
          if (x0pos) { c[idx] = a[idx] + d[idx] * d[idx]; } else { c[idx] += e[idx] * e[idx]; }
        }
      }
    }
  } else {
    for (int64_t i = 0; i < n32; i += 32) {
      for (int64_t k = 0; k < 32; ++k) {
        const int64_t idx = i + k;
        if (a[idx] > b[idx]) {
          a[idx] += b[idx] * d[idx];
          c[idx] = d[idx] * e[idx] + 1.0;
        } else {
          b[idx] = a[idx] + e[idx] * e[idx];
          if (x0pos) { c[idx] = a[idx] + d[idx] * d[idx]; } else { c[idx] += e[idx] * e[idx]; }
        }
      }
    }
  }
  for (int64_t i = n32; i < LEN_1D; ++i) {
    if (a[i] > b[i]) {
      a[i] += b[i] * d[i];
      if (LEN_1D > 10) c[i] += d[i] * d[i]; else c[i] = d[i] * e[i] + 1.0;
    } else {
      b[i] = a[i] + e[i] * e[i];
      if (x0pos) c[i] = a[i] + d[i] * d[i]; else c[i] += e[i] * e[i];
    }
  }
}
