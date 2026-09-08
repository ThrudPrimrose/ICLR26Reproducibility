#include <stdint.h>
#include <omp.h>

#define DEV_CHUNK 32768

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;

  const int big = LEN_1D > 10;
  const int pos = x[0] > 0.0;
  const int64_t n_dev = big ? (DEV_CHUNK < LEN_1D ? DEV_CHUNK : LEN_1D) : 0;

  if (n_dev > 0) {
    #pragma omp target map(to: d[0:n_dev], e[0:n_dev]) \
        map(tofrom: a[0:n_dev], b[0:n_dev], c[0:n_dev])
    {
      for (int64_t i = 0; i < n_dev; ++i) {
        const int gt = a[i] > b[i];
        const double d2 = d[i] * d[i];
        const double e2 = e[i] * e[i];
        const double an = a[i] + b[i] * d[i];
        const double bn = a[i] + e2;
        a[i] = gt ? an : a[i];
        b[i] = gt ? b[i] : bn;
        c[i] = gt ? c[i] + d2 : (pos ? a[i] + d2 : c[i] + e2);
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (int64_t i = n_dev; i < LEN_1D; ++i) {
    const int gt = a[i] > b[i];
    const double d2 = d[i] * d[i];
    const double e2 = e[i] * e[i];
    const double an = a[i] + b[i] * d[i];
    const double bn = a[i] + e2;
    a[i] = gt ? an : a[i];
    b[i] = gt ? b[i] : bn;
    if (big) c[i] = gt ? c[i] + d2 : (pos ? a[i] + d2 : c[i] + e2);
    else     c[i] = gt ? d[i] * e[i] + 1.0 : (pos ? a[i] + d2 : c[i] + e2);
  }
}
