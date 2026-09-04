#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c, const double *restrict d,
                       const double *restrict e, const double *restrict x, const int64_t LEN_1D) {

  const int cond_len = LEN_1D > 10;
  const int cond_x = x[0] > 0.0;

  #pragma omp parallel for
  for (int64_t i = 0; i < LEN_1D; ++i) {
    const double ai = a[i];
    const double bi = b[i];
    const double ci = c[i];
    const double di = d[i];
    const double ei = e[i];

    const int mask = ai > bi;

    const double a_new = ai + bi * di;
    const double b_new = ai + ei * ei;

    const double c_true = cond_len ? (ci + di * di) : (di * ei + 1.0);
    const double c_false = cond_x ? (ai + di * di) : (ci + ei * ei);

    a[i] = mask ? a_new : ai;
    b[i] = mask ? bi : b_new;
    c[i] = mask ? c_true : c_false;
  }
}
