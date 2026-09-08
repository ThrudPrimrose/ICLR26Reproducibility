#include <stdint.h>

/* Reference recurrence: x_{i+1}=b[i], y_{i+1}=x_i  =>  x_i = b[i-1], y_i = b[i-2]
   (wrap: x_0 = b[LEN-1], y_0 = b[LEN-2]).
   So a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333  (indices wrap), exactly the same
   FP ops/order as the sequential reference -- bit identical, fully pointwise. */
void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  const double bm2 = b[LEN_1D - 2];
  const double bm1 = b[LEN_1D - 1];
  a[0] = (b[0] + bm1 + bm2) * 0.333;
  int64_t i = 1;
  if (LEN_1D > 1) {
    a[1] = (b[1] + b[0] + bm1) * 0.333;
    i = 2;
  }
  for (; i < LEN_1D; i++) {
    a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333;
  }
}
