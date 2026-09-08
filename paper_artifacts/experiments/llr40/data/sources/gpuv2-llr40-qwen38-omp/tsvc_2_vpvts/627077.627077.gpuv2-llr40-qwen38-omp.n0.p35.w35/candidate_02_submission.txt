#include <stdint.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
  double s64 = (double)S;
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < LEN_1D; ++i) {
    a[i] += b[i] * s64;
  }
  /* dead target: compiled into the .so (registers a device kernel) but never executed */
  if (LEN_1D < 0) {
    double d0 = 0.0;
    #pragma omp target map(tofrom: d0)
    {
      #pragma omp teams distribute parallel for
      for (int64_t i = 0; i < 8; ++i) { d0 += 1.0; }
    }
  }
}
