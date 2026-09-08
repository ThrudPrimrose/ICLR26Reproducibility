#include <stdint.h>
#include <math.h>
#include <omp.h>

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  if (LEN_1D <= 0) {
    out_value[0] = 0.0;
    out_index[0] = 0;
    return;
  }
  double mv = -INFINITY;
  double dv = INFINITY;
  #pragma omp target data map(to: a[0:LEN_1D])
  {
    #pragma omp target teams distribute parallel for simd reduction(max: mv)
    for (int64_t i = 0; i < LEN_1D; ++i) mv = a[i];
    #pragma omp target teams distribute parallel for simd reduction(min: dv)
    for (int64_t i = 0; i < LEN_1D; ++i)
      dv = fmin(dv, (a[i] == mv) ? (double)i : INFINITY);
  }
  out_value[0] = mv;
  out_index[0] = (int64_t)dv;
}
