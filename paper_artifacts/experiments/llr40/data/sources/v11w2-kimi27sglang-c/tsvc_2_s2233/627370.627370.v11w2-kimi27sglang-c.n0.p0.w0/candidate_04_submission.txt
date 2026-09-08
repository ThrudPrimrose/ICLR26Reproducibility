#include <stdint.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {

  const int64_t BS = 512;

  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (int64_t ii = 8; ii < LEN_2D; ii += BS) {
      int64_t iend = ii + BS < LEN_2D ? ii + BS : LEN_2D;
      for (int64_t j = 8; j < LEN_2D; ++j) {
        #pragma omp simd
        for (int64_t i = ii; i < iend; ++i) {
          aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
        }
      }
    }

    #pragma omp for schedule(static)
    for (int64_t jj = 8; jj < LEN_2D; jj += BS) {
      int64_t jend = jj + BS < LEN_2D ? jj + BS : LEN_2D;
      for (int64_t i = 8; i < LEN_2D; ++i) {
        #pragma omp simd
        for (int64_t j = jj; j < jend; ++j) {
          bb[i * LEN_2D + j] = bb[(i - 1) * LEN_2D + j] + cc[i * LEN_2D + j];
        }
      }
    }
  }
}
