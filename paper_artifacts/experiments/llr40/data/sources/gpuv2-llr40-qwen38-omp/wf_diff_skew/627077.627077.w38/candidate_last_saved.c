#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

static double now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return (double)ts.tv_sec*1e9 + ts.tv_nsec; }

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
  (void)LEN_2D;
  int on_dev=0;
#pragma omp target map(from: on_dev)
  on_dev = !omp_is_initial_device();
  printf("on_device=%d num_devices=%d max_threads=%d\n", on_dev, omp_get_num_devices(), omp_get_max_threads());
  /* measure H2D of the actual input array: 64MB chunk */
  size_t n = (size_t)(64u<<20)/8;
  double t0=now_ns();
#pragma omp target update to(a[:n])
  double t1=now_ns();
#pragma omp target update from(a[:n])
  double t2=now_ns();
#pragma omp target update to(a[:n])
  double t3=now_ns();
  printf("H2D=%.0f ns (%.2f GB/s) D2H=%.0f ns (%.2f GB/s) H2D2=%.0f ns (%.2f GB/s)\n",
    t1-t0, (n*8.0)/(t1-t0)/1e9, t2-t1, (n*8.0)/(t2-t1)/1e9, t3-t2, (n*8.0)/(t3-t2)/1e9);
  fflush(stdout);
}
