/* Optimized TSVC tsvc_2 s1232: triangular region aa[i,j] = bb[i,j]+cc[i,j]
 * for i in [j*VLEN, LEN_2D).
 *
 * Interchange j/i: for each row i, valid columns are j in [0, min(LEN_2D,
 * i/VLEN+1)).  Inner loop is then contiguous -> vectorizes; rows are
 * independent -> OpenMP parallel. */
#include <stdint.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                       const int64_t LEN_2D, const int64_t VLEN) {
  if (LEN_2D <= 0)
    return;
  if (VLEN <= 1) {
    /* jmax = i/VLEN + 1, no division needed for VLEN == 1 */
#pragma omp parallel for schedule(static, 4)
    for (int64_t i = 0; i < LEN_2D; ++i) {
      int64_t jmax = i + 1;
      if (jmax > LEN_2D)
        jmax = LEN_2D;
      double *a = aa + i * LEN_2D;
      const double *b = bb + i * LEN_2D;
      const double *c = cc + i * LEN_2D;
#pragma omp simd
      for (int64_t j = 0; j < jmax; ++j)
        a[j] = b[j] + c[j];
    }
  } else {
#pragma omp parallel for schedule(static, 4)
    for (int64_t i = 0; i < LEN_2D; ++i) {
      int64_t jmax = i / VLEN + 1;
      if (jmax > LEN_2D)
        jmax = LEN_2D;
      double *a = aa + i * LEN_2D;
      const double *b = bb + i * LEN_2D;
      const double *c = cc + i * LEN_2D;
#pragma omp simd
      for (int64_t j = 0; j < jmax; ++j)
        a[j] = b[j] + c[j];
    }
  }
}
