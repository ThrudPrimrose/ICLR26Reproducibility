#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c, const double *restrict d,
                       const double *restrict e, const double *restrict x, const int64_t LEN_1D) {

  const int len_gt_10 = LEN_1D > 10;
  const int x_pos = x[0] > 0.0;

  #pragma omp parallel for simd schedule(static)
  for (int64_t i = 0; i < LEN_1D; ++i) {
    const double ai = a[i];
    const double bi = b[i];
    const double ci = c[i];
    const double di = d[i];
    const double ei = e[i];
    const double cond_d = (ai > bi) ? 1.0 : 0.0;
    const double cond_n = 1.0 - cond_d;

    const double dd = di * di;
    const double ee = ei * ei;

    a[i] = ai + cond_d * bi * di;
    b[i] = cond_d * bi + cond_n * (ai + ee);

    const double c_true  = len_gt_10 ? (ci + dd) : (di * ei + 1.0);
    const double c_false = x_pos      ? (ai + dd) : (ci + ee);
    c[i] = cond_d * c_true + cond_n * c_false;
  }
}
