#include <stdint.h>
#include <stdbool.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t CHUNK = 64;

  #pragma omp parallel for schedule(static) if(LEN_2D > 256)
  for (int64_t ii = 0; ii < LEN_2D; ii += CHUNK) {
    int64_t iend = ii + CHUNK;
    if (iend > LEN_2D) iend = LEN_2D;

    bool any = false;
    for (int64_t i = ii; i < iend; i++) {
      if (aa[i] > 0.0) { any = true; break; }
    }
    if (!any) continue;

    for (int64_t j = 1; j < LEN_2D; j++) {
      #pragma omp simd
      for (int64_t i = ii; i < iend; i++) {
        if (aa[i] > 0.0) {
          aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i] * cc[j * LEN_2D + i];
        }
      }
    }
  }
}
