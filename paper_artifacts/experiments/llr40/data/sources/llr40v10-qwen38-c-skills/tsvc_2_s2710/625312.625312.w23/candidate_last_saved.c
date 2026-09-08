#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c, const double *restrict d,
                       const double *restrict e, const double *restrict x, const int64_t LEN_1D) {
  const double x0 = x[0];
  if (LEN_1D > 10) {
    if (x0 > 0.0) {
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < LEN_1D; ++i) {
        const double ai = a[i];
        const double bi = b[i];
        const double di = d[i];
        const double ei = e[i];
        const int m = ai > bi;
        a[i] = m ? ai + bi * di : ai;
        b[i] = m ? bi : ai + ei * ei;
        c[i] = m ? c[i] + di * di : ai + di * di;
      }
    } else {
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < LEN_1D; ++i) {
        const double ai = a[i];
        const double bi = b[i];
        const double di = d[i];
        const double ei = e[i];
        const int m = ai > bi;
        a[i] = m ? ai + bi * di : ai;
        b[i] = m ? bi : ai + ei * ei;
        c[i] = m ? c[i] + di * di : c[i] + ei * ei;
      }
    }
  } else {
    if (x0 > 0.0) {
      for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > b[i]) {
          a[i] += b[i] * d[i];
          c[i] = d[i] * e[i] + 1.0;
        } else {
          b[i] = a[i] + e[i] * e[i];
          c[i] = a[i] + d[i] * d[i];
        }
      }
    } else {
      for (int64_t i = 0; i < LEN_1D; ++i) {
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
