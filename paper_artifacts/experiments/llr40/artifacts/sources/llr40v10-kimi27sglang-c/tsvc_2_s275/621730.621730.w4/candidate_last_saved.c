#include <stdint.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {

  #pragma omp parallel for simd simdlen(8) schedule(static)
  for (int64_t i = 0; i < LEN_2D; i++) {
    if (aa[i] > 0.0) {
      for (int64_t j = 1; j < LEN_2D; j++) {
        aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + bb[j * LEN_2D + i] * cc[j * LEN_2D + i];
      }
    }
  }
}
