#include <stdint.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                      const int64_t LEN_2D, void *workspace, const int64_t workspace_size) {
  int64_t M = LEN_2D * LEN_2D;
  #pragma omp target map(tofrom: aa[0:M]) map(to: bb[0:M]) map(to: cc[0:M])
  #pragma omp parallel for
  for (int64_t i = 0; i < LEN_2D; i++) {
    if (aa[i] > 0.0) {
      double run = aa[i];
      for (int64_t j = 1; j < LEN_2D; j++) {
        run += bb[j*LEN_2D + i] * cc[j*LEN_2D + i];
        aa[j*LEN_2D + i] = run;
      }
    }
  }
}
