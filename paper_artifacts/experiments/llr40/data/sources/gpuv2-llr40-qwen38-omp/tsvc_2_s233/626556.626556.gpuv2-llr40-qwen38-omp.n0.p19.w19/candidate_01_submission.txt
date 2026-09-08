#include <stdint.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  int64_t n2 = LEN_2D * LEN_2D;
  #pragma omp target data map(to: cc[0:n2]) map(tofrom: aa[0:n2]) map(tofrom: bb[0:n2])
  {
    #pragma omp target
    {
      #pragma omp teams distribute parallel for
      for (int64_t i = 8; i < LEN_2D; ++i) {
        for (int64_t j = 8; j < LEN_2D; ++j) {
          aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
        }
      }
    }
    #pragma omp target
    {
      #pragma omp teams distribute parallel for
      for (int64_t j = 8; j < LEN_2D; ++j) {
        for (int64_t i = 8; i < LEN_2D; ++i) {
          bb[j * LEN_2D + i] = bb[j * LEN_2D + (i - 1)] + cc[j * LEN_2D + i];
        }
      }
    }
  }
}
