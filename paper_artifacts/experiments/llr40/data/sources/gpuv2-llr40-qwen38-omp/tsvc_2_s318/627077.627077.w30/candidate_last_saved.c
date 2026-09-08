#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D, const int64_t inc) {
  if (LEN_1D <= 0) { result[0] = 0.0; return; }
  const int64_t nelem = (LEN_1D == 1 || inc == 0) ? 1 : ((LEN_1D - 1) * inc + 1);

  #pragma omp target map(to : a[(int64_t)nelem]) map(from: result[1])
  {
    double maxv = fabs(a[0]);
    int64_t index = 0;
    for (int64_t i = 1; i < LEN_1D; ++i) {
      double v = fabs(a[i * inc]);
      if (v > maxv) { index = i; maxv = v; }
    }
    result[0] = maxv + (double)index;
  }
}
