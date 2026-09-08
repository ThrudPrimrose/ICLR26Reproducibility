#include <stdint.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  #pragma omp parallel
  {
    for (int64_t k = 8; k < LEN_2D; ++k) {
      #pragma omp for nowait
      for (int64_t idx = 8; idx < LEN_2D; ++idx) {
        aa[k * LEN_2D + idx] = aa[(k - 1) * LEN_2D + idx] + cc[k * LEN_2D + idx];
        bb[k * LEN_2D + idx] = bb[(k - 1) * LEN_2D + idx] + cc[k * LEN_2D + idx];
      }
    }
  }
}
