#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
  const int64_t n = LEN_1D - 1;
  if (n <= 0) return;
  double t0 = 0, t1 = 0;
  int show = 0;
  {
    show = 1;
    {
      const char *env = getenv("OMP_NUM_THREADS");
      printf("TSHOW LEN_1D=%lld max_threads=%d env_OMP_NUM_THREADS=%s\n", (long long)LEN_1D, omp_get_max_threads(), env ? env : "(null)");
      fflush(stdout);
    }
    t0 = omp_get_wtime();
  }
  a[0] = b[0] + c[0] * c[0] + b[0] * b[0] + c[0];
#pragma omp parallel for schedule(static)
  for (int64_t j = 1; j < n; j++) {
    const double oldj = a[j];
    a[j] = b[j] + c[j] * c[j] + b[j] * b[j] + c[j];
    d[j - 1] = b[j - 1] + c[j - 1] * c[j - 1] + b[j - 1] * b[j - 1] + c[j - 1] + oldj;
  }
  d[n - 1] = a[n - 1] + a[n];
  if (show) {
    t1 = omp_get_wtime();
    printf("TSHOW ms=%.4f GBs_min=%.1f\n", (t1 - t0) * 1e3, (double)LEN_1D * 8.0 * 5.0 / (t1 - t0) / 1e9);
    fflush(stdout);
  }
}
