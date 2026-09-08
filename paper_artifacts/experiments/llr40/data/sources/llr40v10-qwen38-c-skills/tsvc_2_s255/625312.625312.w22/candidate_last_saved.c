#include <stdint.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  /* a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333, indices wrap for i=0,1.
   * The reference carries x=b[i-1], y=b[i-2] as scalars; substituting the
   * expression back in removes the carried dependence entirely, so the loop
   * is parallel and vectorizable. Only i=0,1 need the wraparound. */
  a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * 0.333;
  if (LEN_1D > 1)
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * 0.333;

  #pragma omp parallel for simd schedule(static)
  for (int64_t i = 2; i < LEN_1D; i++) {
    a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333;
  }
}
