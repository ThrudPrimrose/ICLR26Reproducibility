#include <stdint.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  #pragma omp target teams distribute parallel for \
    map(to: a[0:LEN_1D]) map(from: out[0:LEN_1D])
  for (int64_t i = 0; i < LEN_1D; ++i) {
    double t = a[i] * a[i];
    out[i] = (t + 1.0) * (t - 1.0);
  }
}
