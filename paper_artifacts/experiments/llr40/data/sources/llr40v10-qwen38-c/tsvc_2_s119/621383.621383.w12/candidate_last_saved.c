#include <stdint.h>
#include <omp.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  if (LEN_2D < 2) return;
  #pragma omp parallel
  {
    for (int64_t i = 1; i < LEN_2D; ++i) {
      #pragma omp for
      for (int64_t j = 1; j < LEN_2D; ++j) {
        aa[i * LEN_2D + j] = aa[(i - 1) * LEN_2D + (j - 1)] + bb[i * LEN_2D + j];
      }
    }
  }
}
