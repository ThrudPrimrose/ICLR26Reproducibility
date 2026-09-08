#include <stdint.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b,
                      const double *restrict c, const int64_t LEN_1D) {
  #pragma STDC FP_CONTRACT OFF
  #pragma omp target map(to: b[0:LEN_1D], c[0:LEN_1D]) map(from: a[0:LEN_1D])
  #pragma omp teams distribute parallel for
  for (int64_t i = 0; i < LEN_1D; ++i) {
    double s = b[i] * c[i];
    double t = (i > 0) ? b[i - 1] * c[i - 1] : 0.0;
    a[i] = s + t;
  }
}
