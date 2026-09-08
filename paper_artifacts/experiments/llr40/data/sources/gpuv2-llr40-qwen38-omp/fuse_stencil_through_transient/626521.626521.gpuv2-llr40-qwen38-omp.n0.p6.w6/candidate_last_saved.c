#include <stdint.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  if (LEN_1D > 300000) {
    #pragma omp target teams distribute parallel for map(to: a[0:LEN_1D]) map(from: out[1:LEN_1D - 3])
    for (int64_t i = 1; i < LEN_1D - 2; ++i) {
      out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
    return;
  }
  if (LEN_1D > 32768) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < LEN_1D - 2; ++i) {
      out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
    return;
  }
  for (int64_t i = 1; i < LEN_1D - 2; ++i) {
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
  }
}
