/* tsvc_2 s255: "serial" chain x,y where x_i=b[i-1], y_i=b[i-2] (wrap at i=0,1).
 * Unrolled: a[0]=(b[0]+b[N-1]+b[N-2])*0.333, a[1]=(b[1]+b[0]+b[N-1])*0.333,
 * a[i]=(b[i]+b[i-1]+b[i-2])*0.333.  Bit-identical FP op order to the reference.
 * Offloaded to GPU via OpenMP target. */
#include <stdint.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  if (LEN_1D == 1) {
    a[0] = (b[0] + b[0] + b[0]) * 0.333;
    return;
  }
  #pragma omp target map(to: b[0:LEN_1D]) map(from: a[0:LEN_1D])
  {
    #pragma omp parallel for
    for (int64_t i = 2; i < LEN_1D; i++)
      a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333;
    a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * 0.333;
    a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * 0.333;
  }
}
