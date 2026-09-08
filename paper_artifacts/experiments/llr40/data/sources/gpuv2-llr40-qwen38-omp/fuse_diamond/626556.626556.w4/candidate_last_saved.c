#include <stdint.h>
#include <stdio.h>
#include <omp.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  #pragma omp target teams distribute parallel for map(to: a[0:LEN_1D]) map(from: out[0:LEN_1D])
  for (int64_t i = 0; i < LEN_1D; ++i) {
    double t = a[i] * a[i];
    out[i] = (t + 1.0) * (t - 1.0);
  }
  int dev = 0;
  #pragma omp target map(from: dev)
  { dev = !omp_is_initial_device(); }
  printf("PROBE LEN_1D=%lld dev=%d\n", (long long)LEN_1D, dev);
  fflush(stdout);
}
