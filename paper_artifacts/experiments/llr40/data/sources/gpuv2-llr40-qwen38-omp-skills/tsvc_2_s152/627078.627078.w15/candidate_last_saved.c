#include <stdint.h>

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e, const int64_t LEN_1D) {
  #pragma omp target data map(to: c[0:LEN_1D], d[0:LEN_1D], e[0:LEN_1D]) \
                          map(tofrom: a[0:LEN_1D], b[0:LEN_1D])
  {
    #pragma omp target teams distribute parallel for simd
    for (int64_t i = 0; i < LEN_1D; ++i) {
      const double bi = d[i] * e[i];
      b[i] = bi;
      a[i] += bi * c[i];
    }
  }
}
