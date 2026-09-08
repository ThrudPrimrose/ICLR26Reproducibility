#include <math.h>
#include <stdint.h>
#include <omp.h>

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc) {
  if (LEN_1D <= 0) { result[0] = 0.0; return; }

  int64_t span = (LEN_1D > 1 && inc > 0) ? (LEN_1D - 1) * inc + 1 : 1;

  if (span < (1LL << 19)) {  /* small: serial host */
    double maxv = fabs(a[0]);
    int64_t index = 0;
    for (int64_t i = 1; i < LEN_1D; ++i) {
      double v = fabs(a[i * inc]);
      if (v > maxv) { maxv = v; index = i; }
    }
    result[0] = maxv + (double)index;
    return;
  }

  double maxv = -1.0;       /* identity for max over fabs values */
  int64_t first = INT64_MAX;
  #pragma omp target data map(to: a[0:span])
  {
    #pragma omp target teams distribute parallel for reduction(max: maxv)
    for (int64_t i = 0; i < LEN_1D; ++i) {
      double v = fabs(a[i * inc]);
      if (v > maxv) maxv = v;
    }
    #pragma omp target teams distribute parallel for reduction(min: first)
    for (int64_t i = 0; i < LEN_1D; ++i) {
      if (fabs(a[i * inc]) == maxv && i < first) first = i;
    }
  }
  result[0] = maxv + (double)first;
}
