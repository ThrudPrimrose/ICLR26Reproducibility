#include <stdint.h>
#include <math.h>
#include <omp.h>

typedef struct { double val; int64_t idx; } pair;

#pragma omp declare reduction(maxidx : pair : \
    omp_out = (omp_in.val > omp_out.val || (omp_in.val == omp_out.val && omp_in.idx < omp_out.idx)) ? omp_in : omp_out) \
    initializer(omp_priv = {-1.0/0.0, INT64_MAX})

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  pair best = {-1.0/0.0, INT64_MAX};
  #pragma omp parallel for simd reduction(maxidx : best) schedule(static)
  for (int64_t i = 0; i < LEN_1D; ++i) {
    if (a[i] > best.val) {
      best.val = a[i];
      best.idx = i;
    }
  }
  out_value[0] = best.val;
  out_index[0] = best.idx;
}
