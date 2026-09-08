#include <stdint.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D,
                       const int64_t VLEN) {
  const int64_t n = LEN_2D * LEN_2D;
  #pragma omp target map(tofrom: aa[0:n]) map(to: bb[0:n]) map(to: cc[0:n])
  {
    #pragma omp teams distribute parallel for schedule(dynamic, 16)
    for (int64_t i = 0; i < LEN_2D; ++i) {
      const int64_t w = i / VLEN + 1;
      for (int64_t j = 0; j < w; ++j) {
        aa[i * LEN_2D + j] = bb[i * LEN_2D + j] + cc[i * LEN_2D + j];
      }
    }
  }
}
