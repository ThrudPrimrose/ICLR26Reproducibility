#include <stdint.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  const int64_t hi = LEN_1D - 2;
  if (hi <= 1) return;
  if (hi - 1 <= 32768) {
    for (int64_t i = 1; i < hi; ++i) {
      out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
    return;
  }
  #pragma omp parallel for simd schedule(static)
  for (int64_t i = 1; i < hi; ++i) {
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
  }
}
