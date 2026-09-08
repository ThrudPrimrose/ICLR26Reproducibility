#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
  const int64_t n2 = LEN_2D * (int64_t)LEN_2D;
  const int64_t n2g = n2 / 6;                 /* gpu slice: tail of array */
  const double *base = aa + (n2 - n2g);
  static double pm[1024];
  static int64_t pi[1024];
  int devnt = 0;
  double t0 = omp_get_wtime();
  double tm = -1.0;
  #pragma omp target map(to: base[0:n2g]) map(from: pm[0:1024], pi[0:1024], devnt)
  {
    double dw0 = omp_get_wtime();
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      double lm = -INFINITY;
      int64_t li = -1;
      #pragma omp for schedule(static)
      for (int64_t k = 0; k < n2g; ++k) {
        double v = base[k];
        if (v > lm) { lm = v; li = k; }
      }
      pm[tid] = lm;
      pi[tid] = li;
      if (tid == 0) { devnt = omp_get_num_threads(); tm = omp_get_wtime() - dw0; }
    }
  }
  double t1 = omp_get_wtime();
  double maxv = -INFINITY;
  int64_t xindex = -1;
  for (int t = 0; t < 1024; ++t) {
    double v = pm[t];
    if (v > maxv) { maxv = v; xindex = pi[t]; }
    else if (v == maxv && pi[t] < xindex) { xindex = pi[t]; }
  }
  if (xindex < 0) xindex = 0;
  bb[0] = maxv + (double)(xindex + (n2 - n2g));
  double t2 = omp_get_wtime();
  printf("devnt=%d transfer+scan_hostwall=%.2f ms dev_scan=%.2f ms gather=%.2f ms\n",
         devnt, (t1 - t0) * 1e3, tm * 1e3, (t2 - t1) * 1e3);
  fflush(stdout);
}
