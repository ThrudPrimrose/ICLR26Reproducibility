#include <stdint.h>
#include <omp.h>

void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  double acc = 0.0;
  #pragma omp parallel for simd reduction(+:acc)
  for (int64_t i = 0; i < LEN_1D; i++) {
    acc += (i & 1) ? a[i] : 0.0;
  }
  out[0] = acc;
}
