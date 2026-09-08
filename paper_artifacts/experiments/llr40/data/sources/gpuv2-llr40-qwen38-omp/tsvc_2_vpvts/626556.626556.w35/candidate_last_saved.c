#include <stdint.h>
#include <omp.h>
#include <stdio.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
  if (LEN_1D <= 0) return;
  const double s = (double)S;
  static int probed = 0;
  if (!probed) {
    int on_device = 0;
    #pragma omp target map(from: on_device)
    on_device = !omp_is_initial_device();
    probed = 1;
    fprintf(stderr, "tsvc_2_vpvts: running on device=%d (1=GPU)\n", on_device);
  }
  #pragma omp target teams distribute parallel for map(tofrom: a[0:LEN_1D]) map(to: b[0:LEN_1D], s)
  for (int64_t i = 0; i < LEN_1D; ++i) {
    a[i] += b[i] * s;
  }
}
