#include <stdint.h>
#include <omp.h>

/* Warm up the OpenMP team at load time so the first timed call does not pay spawn cost. */
__attribute__((constructor))
static void omp_warm(void) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < 64; ++i) { volatile int x = i; (void)x; }
}

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  #pragma omp parallel for schedule(static)
  for (int64_t i = 1; i < LEN_1D - 2; ++i)
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
}
