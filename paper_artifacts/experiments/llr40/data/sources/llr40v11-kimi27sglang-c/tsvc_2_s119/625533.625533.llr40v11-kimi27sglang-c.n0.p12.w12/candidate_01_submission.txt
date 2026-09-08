#include <stdint.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {

  #pragma omp parallel
  {
    for (int64_t i = 1; i < LEN_2D; ++i) {
      double *restrict a_row = aa + i * LEN_2D;
      const double *restrict a_prev = aa + (i - 1) * LEN_2D;
      const double *restrict b_row = bb + i * LEN_2D;
      #pragma omp for simd schedule(static)
      for (int64_t j = 1; j < LEN_2D; ++j) {
        a_row[j] = a_prev[j - 1] + b_row[j];
      }
    }
  }
}
