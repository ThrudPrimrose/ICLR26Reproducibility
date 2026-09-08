#include <omp.h>
#include <stdint.h>

/* Full GPU offload variant: MI300A CPU and GPU share HBM3, so the map round trip
 * is an on-package copy, not a PCIe transfer.  The CDNA3 device streams the five
 * arrays far faster than 12 Zen4 cores do.
 * sum: per device-thread partials (atomic capture into 1024 slots), reduced on host. */

#define PARTS 1024

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e, const int64_t LEN_1D) {
  double sum = 0.0;

  double *partials;
#pragma omp target
  partials = (double *)omp_target_alloc(PARTS * sizeof(double), 0);

#pragma omp target data map(to: c[0:LEN_1D], d[0:LEN_1D], e[0:LEN_1D]) \
                        map(from: a[0:LEN_1D], b[0:LEN_1D]) \
                        map(to: partials[0:PARTS])
  {
#pragma omp target teams distribute
    for (int64_t r = 0; r < PARTS; ++r) partials[r] = 0.0;

#pragma omp target teams distribute parallel for
    for (int64_t i = 0; i < LEN_1D; ++i) {
      double ai = c[i] + d[i];
      double bi = c[i] + e[i];
      a[i] = ai;
      b[i] = bi;
      double old;
#pragma omp atomic capture
      old = partials[omp_get_thread_num() & (PARTS - 1)] += ai + bi;
      (void)old;
    }
  }

#pragma omp target data map(from: partials[0:PARTS])
  { }
  for (int r = 0; r < PARTS; ++r) sum += partials[r];

#pragma omp target
  omp_target_free(partials, 0);

  b[0] = sum;
}
