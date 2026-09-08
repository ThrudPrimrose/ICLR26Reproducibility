#include <omp.h>
#include <stdint.h>
#include <stdio.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
  int ondev = 0;
  #pragma omp target map(to: a[:1]) map(from: ondev)
  { a[0] = a[0] + 0.0; ondev = !omp_is_initial_device(); }
  printf("LEN_2D=%lld ondev=%d\n", (long long)LEN_2D, ondev);
  fflush(stdout);
}
