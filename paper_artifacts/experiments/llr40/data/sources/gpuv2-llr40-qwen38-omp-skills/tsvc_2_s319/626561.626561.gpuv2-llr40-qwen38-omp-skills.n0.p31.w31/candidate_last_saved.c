#include <stdint.h>
#include <omp.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  double sum = 0.0;

  #pragma omp target teams distribute parallel for reduction(+:sum) \
      map(to: c[0:LEN_1D]) map(to: d[0:LEN_1D]) map(to: e[0:LEN_1D]) \
      map(from: a[0:LEN_1D]) map(from: b[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; ++i) {
      a[i] = c[i] + d[i];
      sum += a[i];
      b[i] = c[i] + e[i];
      sum += b[i];
    }
  b[0] = sum;
}
