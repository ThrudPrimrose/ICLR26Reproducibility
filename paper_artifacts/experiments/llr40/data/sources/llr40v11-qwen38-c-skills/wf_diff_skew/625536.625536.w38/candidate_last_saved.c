#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n < 2) return;
  for (int64_t i = 1; i < n; ++i) {
    const double *restrict p = a + (i - 1) * n;
    double *restrict q = a + i * n;
    for (int64_t j = 0; j < n - 1; ++j)
      q[j] = q[j] + p[j] + p[j + 1];
  }
  fprintf(stdout, "diag n=%ld T=%ld\n", (long)n, (long)omp_get_max_threads());
  fflush(stdout);
}
