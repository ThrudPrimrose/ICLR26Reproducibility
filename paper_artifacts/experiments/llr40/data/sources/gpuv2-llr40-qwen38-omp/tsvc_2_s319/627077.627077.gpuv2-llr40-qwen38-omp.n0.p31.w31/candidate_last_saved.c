#include <stdint.h>
#include <stdio.h>
#include <omp.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  double sum = 0.0;

  #pragma omp target teams distribute parallel for map(to: c[0:n], d[0:n], e[0:n]) map(from: a[0:n], b[0:n]) reduction(+:sum)
  for (int64_t i = 0; i < n; ++i)
  {
    double ai = c[i] + d[i];
    a[i] = ai;
    sum += ai;
    double bi = c[i] + e[i];
    b[i] = bi;
    sum += bi;
  }
  double t0 = omp_get_wtime();
  #pragma omp target teams distribute parallel for map(to: c[0:n], d[0:n], e[0:n]) map(from: a[0:n], b[0:n]) reduction(+:sum)
  for (int64_t i = 0; i < n; ++i)
  {
    double ai = c[i] + d[i];
    a[i] = ai;
    sum += ai;
    double bi = c[i] + e[i];
    b[i] = bi;
    sum += bi;
  }
  double t1 = omp_get_wtime();
  #pragma omp target data map(to: c[0:n], d[0:n], e[0:n])
  {
  }
  double t2 = omp_get_wtime();
  #pragma omp target data map(from: a[0:n], b[0:n])
  {
  }
  double t3 = omp_get_wtime();

  b[0] = sum;
  printf("STEADY FULL=%.3f H2D=%.3f D2H=%.3f\n", (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3);
  fflush(stdout);
}
