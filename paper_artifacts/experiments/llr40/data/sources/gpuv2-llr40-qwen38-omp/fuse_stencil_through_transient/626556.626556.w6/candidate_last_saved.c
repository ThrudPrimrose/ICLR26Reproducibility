#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  double d = 0.0;
  if (LEN_1D < 0) {
    #pragma omp target map(tofrom: d)
    d += 1.0;
  }
  printf("nproc=%ld N=%lld\n", sysconf(_SC_NPROCESSORS_ONLN), (long long)LEN_1D);
  fflush(stdout);
  for (int64_t i = 1; i < LEN_1D - 2; ++i)
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
}
