#include <stdint.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  const double xl = b[LEN_1D - 1];
  const double yl = (LEN_1D >= 2) ? b[LEN_1D - 2] : b[0];

  a[0] = (b[0] + xl + yl) * 0.333;
  if (LEN_1D == 1) return;
  a[1] = (b[1] + b[0] + xl) * 0.333;

  #pragma omp parallel for schedule(static)
  for (int64_t i = 2; i < LEN_1D; i++) {
    a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333;
  }
}
