#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  const int64_t L = LEN_2D;
  if (L < 2) return;

  const int64_t W = 8;                      /* columns per block = one ZMM lane group */
  const int64_t nblk = (L + W - 1) / W;

  #pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < nblk; ++b) {
    const int64_t i0 = b * W;
    const int64_t w  = i0 + W < L ? W : L - i0;
    double r[8];
    for (int64_t k = 0; k < w; ++k) r[k] = aa[i0 + k];
    for (int64_t j = 1; j < L; ++j) {
      const double *const brow = bb + j * L + i0;
      double *const arow       = aa + j * L + i0;
      #pragma omp simd
      for (int64_t k = 0; k < w; ++k) {
        r[k] += brow[k];
        arow[k] = r[k];
      }
    }
  }
}
