#include <stdint.h>
#include <stdio.h>
#include <omp.h>

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
  double s = 0.0;
  printf("LEN_1D = %lld (%.2f MB)\n", (long long)LEN_1D, 8.0*LEN_1D/(1024.0*1024.0));
  printf("max_threads=%d  threads_in_team=%d\n", omp_get_max_threads(), 1);
  const double t0 = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp for schedule(static) reduction(+:s)
    for (int64_t i = 0; i < LEN_1D; i++) s += a[i];
    if (omp_get_thread_num() == 0) {
      printf("team_size=%d\n", omp_get_num_threads());
    }
  }
  const double t1 = omp_get_wtime();
  sum_out[0] = s;
  double dt = (t1-t0)*1e9;
  printf("kernel_ms=%.3f  gbs=%.1f  sum=%.9e\n", dt/1e6, 8.0*LEN_1D/dt, s);
  /* single-thread vectorized rate */
  {
    double s1=0.0;
    double ts = omp_get_wtime();
    for (int64_t i = 0; i < LEN_1D; i++) s1 += a[i];
    double te = omp_get_wtime();
    printf("serial_ms=%.3f  serial_gbs=%.1f\n", (te-ts)*1e3, 8.0*LEN_1D/(te-ts)/1e9);
  }
  fflush(stdout);
}
