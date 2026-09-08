#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

static double stamp(void){
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9*ts.tv_nsec;
}

static double pass_mt(const double *a, int64_t n, int nt){
  double sum = 0.0;
  #pragma omp parallel num_threads(nt) reduction(+:sum)
  {
    int tid = omp_get_thread_num();
    int64_t per = (n + nt - 1)/nt;
    int64_t lo = (int64_t)tid*per, hi = lo + per; if (hi > n) hi = n;
    const double *p = a + lo;
    for (int64_t i = 0; i < hi - lo; i += 8) {
      __m512d v = _mm512_loadu_pd(p + i);
      v = _mm512_add_pd(v, _mm512_set1_pd(1.0e-300));
      __m256d vl = _mm512_extractf64x4_pd(v, 0), vh = _mm512_extractf64x4_pd(v, 1);
      sum += _mm_cvtsd_f64(_mm256_castpd256_pd128(vl)) + _mm_cvtsd_f64(_mm256_castpd256_pd128(vh));
    }
  }
  return sum;
}

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc) {
  static int done = 0;
  if (!done) {
    done = 1;
    int64_t n = LEN_1D;
    double bytes = (double)n * 8;
    printf("PROBE n=%lld inc=%lld\n", (long long)n, (long long)inc);
    for (int t = 0; t < 8; t++) {
      double t0 = stamp();
      volatile double s = pass_mt(a, n, 24);
      (void)s;
      double t1 = stamp();
      printf("pass mt24 t=%d  %.3f ms  %.1f GB/s\n", t, (t1-t0)*1e3, bytes/(t1-t0)/1e9);
    }
    fflush(stdout);
  }
  result[0] = 0.0;
}
