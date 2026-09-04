/* TSVC s2710: branchy elementwise update.
 * - Unswitched on the two loop-invariant tests (LEN_1D>10, x[0]>0)
 * - Data branch kept as ternary selects -> vectorizes to 64-byte (AVX-512) vectors
 * - Whole loop threads via OpenMP static schedule
 * - fp-contract=off keeps bit-identical IEEE results vs the strict reference
 */
#include <stdint.h>

#pragma GCC optimize("fp-contract=off")

static void core_pos(int64_t n, double *a, double *b, double *c, const double *d, const double *e)
{
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) {
    double ai = a[i], bi = b[i], di = d[i], ei = e[i];
    int t = ai > bi;
    a[i] = t ? ai + bi * di : ai;
    b[i] = t ? bi : ai + ei * ei;
    c[i] = t ? c[i] + di * di : ai + di * di;
  }
}

static void core_neg(int64_t n, double *a, double *b, double *c, const double *d, const double *e)
{
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) {
    double ai = a[i], bi = b[i], di = d[i], ei = e[i];
    int t = ai > bi;
    a[i] = t ? ai + bi * di : ai;
    b[i] = t ? bi : ai + ei * ei;
    c[i] = t ? c[i] + di * di : c[i] + ei * ei;
  }
}

static void core_small_pos(int64_t n, double *a, double *b, double *c, const double *d, const double *e)
{
  for (int64_t i = 0; i < n; ++i) {
    double ai = a[i], bi = b[i], di = d[i], ei = e[i];
    int t = ai > bi;
    a[i] = t ? ai + bi * di : ai;
    b[i] = t ? bi : ai + ei * ei;
    c[i] = t ? di * ei + 1.0 : ai + di * di;
  }
}

static void core_small_neg(int64_t n, double *a, double *b, double *c, const double *d, const double *e)
{
  for (int64_t i = 0; i < n; ++i) {
    double ai = a[i], bi = b[i], di = d[i], ei = e[i];
    int t = ai > bi;
    a[i] = t ? ai + bi * di : ai;
    b[i] = t ? bi : ai + ei * ei;
    c[i] = t ? di * ei + 1.0 : c[i] + ei * ei;
  }
}

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D)
{
  const int64_t n = LEN_1D;
  if (n <= 10) {
    if (x[0] > 0.0)
      core_small_pos(n, a, b, c, d, e);
    else
      core_small_neg(n, a, b, c, d, e);
  } else if (x[0] > 0.0) {
    core_pos(n, a, b, c, d, e);
  } else {
    core_neg(n, a, b, c, d, e);
  }
}
