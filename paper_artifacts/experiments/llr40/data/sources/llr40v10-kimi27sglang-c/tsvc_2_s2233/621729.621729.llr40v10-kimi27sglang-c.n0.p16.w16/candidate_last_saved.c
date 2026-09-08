#include <stdint.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {

  #pragma omp parallel
  {
    for (int64_t r = 8; r < LEN_2D; ++r) {
      #pragma omp for simd schedule(static)
      for (int64_t k = 8; k < LEN_2D; ++k) {
        const double c = cc[r * LEN_2D + k];
        aa[r * LEN_2D + k] = aa[(r - 1) * LEN_2D + k] + c;
        bb[r * LEN_2D + k] = bb[(r - 1) * LEN_2D + k] + c;
      }
    }
  }
}
