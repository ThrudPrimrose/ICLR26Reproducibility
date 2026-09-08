#include <stdint.h>
#include <omp.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
  int64_t nth = omp_get_max_threads();
  int64_t chunk = (LEN_1D + nth - 1) / nth;
  #pragma omp parallel for schedule(static, chunk)
  for (int64_t i = 0; i < LEN_1D; ++i) {
    a[i] = a[i] * b[i] * c[i];
  }
}
