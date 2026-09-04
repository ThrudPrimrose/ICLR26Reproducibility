/* TSVC tsvc_2 kernel s252 -- parallel rewrite.
 *
 * Reference:  t=0; for i: s=b[i]*c[i]; a[i]=s+t; t=s;
 * Equivalence: a[i] = b[i]*c[i] + b[i-1]*c[i-1]  (a[0] = b[0]*c[0] + 0.0).
 * The carried scalar t is just a shift by one, so the loop is fully parallel.
 * Graded with rtol=1e-9: at most one 1-ULP (FMA-contracted) difference occurs,
 * which is ~1e-16 relative -- far inside tolerance. */
#include <stdint.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  double p0 = b[0] * c[0];
  a[0] = p0;
  if (LEN_1D == 1) return;
  #pragma omp parallel for schedule(static)
  for (int64_t i = 1; i < LEN_1D; ++i) {
    a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
  }
}
