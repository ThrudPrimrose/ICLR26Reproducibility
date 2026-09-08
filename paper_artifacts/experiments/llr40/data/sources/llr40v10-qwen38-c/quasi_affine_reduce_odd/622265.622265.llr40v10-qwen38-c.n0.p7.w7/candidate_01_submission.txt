#include <stdint.h>
#include <omp.h>

void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  double total = 0.0;
  const int64_t m = LEN_1D / 2;
  #pragma omp parallel for schedule(static) reduction(+:total)
  for (int64_t j = 0; j < m; j++) total += a[2*j+1];
  out[0] = total;
}
