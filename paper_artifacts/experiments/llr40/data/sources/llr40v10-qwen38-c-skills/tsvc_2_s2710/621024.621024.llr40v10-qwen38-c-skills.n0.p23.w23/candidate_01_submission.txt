/* TSVC_2 s2710 -- if/else with two loop-invariant inner conditions (LEN_1D>10,
 * x[0]>0.0) and one data-dependent outer condition (a[i] > b[i]).
 *
 * Dependences: every read is at index i; no element touches another index, so the
 * loop is fully parallel per element. The two invariant guards are hoisted out
 * (unswitching) so the hot loops are branch-free; the data-dependent guard
 * becomes arithmetic (scalar select), which the vectorizer turns into blends.
 */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D)
{
  const int64_t n = LEN_1D;
  const int big = n > 10;
  const int xpos = x[0] > 0.0;

  if (big && xpos) {
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < n; ++i) {
      const double a0 = a[i], b0 = b[i], c0 = c[i], d0 = d[i], e0 = e[i];
      const int m = a0 > b0;
      const double dd = d0 * d0;
      a[i] = m ? a0 + b0 * d0 : a0;
      b[i] = m ? b0 : a0 + e0 * e0;
      c[i] = m ? c0 + dd : a0 + dd;
    }
  } else if (big) {
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < n; ++i) {
      const double a0 = a[i], b0 = b[i], c0 = c[i], d0 = d[i], e0 = e[i];
      const int m = a0 > b0;
      const double ee = e0 * e0;
      a[i] = m ? a0 + b0 * d0 : a0;
      b[i] = m ? b0 : a0 + ee;
      c[i] = m ? c0 + d0 * d0 : c0 + ee;
    }
  } else {
    /* n <= 10: keep the reference shape, still branch-free. */
    #pragma omp simd
    for (int64_t i = 0; i < n; ++i) {
      const double a0 = a[i], b0 = b[i], c0 = c[i], d0 = d[i], e0 = e[i];
      const int m = a0 > b0;
      const double ee = e0 * e0;
      a[i] = m ? a0 + b0 * d0 : a0;
      b[i] = m ? b0 : a0 + ee;
      c[i] = m ? d0 * e0 + 1.0 : (xpos ? a0 + d0 * d0 : c0 + ee);
    }
  }
}
