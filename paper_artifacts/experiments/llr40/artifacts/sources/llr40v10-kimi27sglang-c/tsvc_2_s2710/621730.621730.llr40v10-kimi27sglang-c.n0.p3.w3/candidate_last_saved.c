#include <stdint.h>
#include <immintrin.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c, const double *restrict d,
                       const double *restrict e, const double *restrict x, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;

  const int len_gt_10 = LEN_1D > 10;
  const int x_pos = x[0] > 0.0;

  #pragma omp parallel for
  for (int64_t i = 0; i < LEN_1D; ++i) {
    _mm_prefetch((const char *)(a + i + 64), _MM_HINT_T0);
    _mm_prefetch((const char *)(b + i + 64), _MM_HINT_T0);
    _mm_prefetch((const char *)(c + i + 64), _MM_HINT_T0);
    _mm_prefetch((const char *)(d + i + 64), _MM_HINT_T0);
    _mm_prefetch((const char *)(e + i + 64), _MM_HINT_T0);

    const double ai = a[i];
    const double bi = b[i];
    const double di = d[i];
    const double ei = e[i];
    const double ci = c[i];

    const int gt = ai > bi;

    const double new_a = gt ? (ai + bi * di) : ai;
    const double new_b = gt ? bi : (ai + ei * ei);
    const double c_gt  = len_gt_10 ? (ci + di * di) : (di * ei + 1.0);
    const double c_le  = x_pos     ? (ai + di * di) : (ci + ei * ei);
    const double new_c = gt ? c_gt : c_le;

    a[i] = new_a;
    b[i] = new_b;
    c[i] = new_c;
  }
}
