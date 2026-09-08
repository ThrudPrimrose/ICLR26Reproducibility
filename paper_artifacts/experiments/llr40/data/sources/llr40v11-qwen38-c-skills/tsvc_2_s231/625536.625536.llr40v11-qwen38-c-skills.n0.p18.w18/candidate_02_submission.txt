#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  if (LEN_2D <= 1) return;

  const int64_t nthreads = omp_get_max_threads();
  int64_t B = (LEN_2D + nthreads - 1) / nthreads;
  B = (B + 7) / 8 * 8;
  if (B < 8) B = 8;
  if (B > LEN_2D) B = LEN_2D;
  const int64_t nblk = (LEN_2D + B - 1) / B;

  #pragma omp parallel for schedule(static, 1)
  for (int64_t b = 0; b < nblk; b++) {
    const int64_t i0 = b * B;
    const int64_t len = (b + 1) * B <= LEN_2D ? B : LEN_2D - i0;
    double *restrict a_prev = aa + i0;
    double *restrict a_cur = aa + LEN_2D + i0;
    const double *restrict b_ptr = bb + LEN_2D + i0;
    int64_t j = 1;
    for (; j + 1 < LEN_2D; j += 2) {
      double *restrict a_cur2 = a_cur + LEN_2D;
      const double *restrict b_ptr2 = b_ptr + LEN_2D;
      for (int64_t i = 0; i < len; i++)
        a_cur[i] = a_prev[i] + b_ptr[i];
      for (int64_t i = 0; i < len; i++)
        a_cur2[i] = a_cur[i] + b_ptr2[i];
      a_prev = a_cur2;
      a_cur += 2 * LEN_2D;
      b_ptr = b_ptr2 + LEN_2D;
    }
    if (j < LEN_2D) {
      for (int64_t i = 0; i < len; i++)
        a_cur[i] = a_prev[i] + b_ptr[i];
    }
  }
}
