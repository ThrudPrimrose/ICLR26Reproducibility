#include <stdint.h>
#include <omp.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
  if (LEN_1D <= (1 << 22)) {
    for (int64_t i = 0; i < LEN_1D; ++i) {
      a[i] = a[i] * b[i] * c[i];
    }
    return;
  }
  #pragma omp parallel for num_threads(24) schedule(static)
  for (int64_t i = 0; i < LEN_1D; ++i) {
    a[i] = a[i] * b[i] * c[i];
  }
}
