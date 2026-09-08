#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec + ts.tv_nsec*1e-9; }

static __inline__ __attribute__((always_inline)) double
vmin32(const double *restrict p, double seed) {
  __m512d m0 = _mm512_set1_pd(seed);
  __m512d m1 = _mm512_set1_pd(seed);
  __m512d m2 = _mm512_set1_pd(seed);
  __m512d m3 = _mm512_set1_pd(seed);
  m0 = _mm512_min_pd(m0, _mm512_loadu_pd(p + 0));
  m1 = _mm512_min_pd(m1, _mm512_loadu_pd(p + 8));
  m2 = _mm512_min_pd(m2, _mm512_loadu_pd(p + 16));
  m3 = _mm512_min_pd(m3, _mm512_loadu_pd(p + 24));
  return _mm512_reduce_min_pd(_mm512_min_pd(_mm512_min_pd(m0, m2),
                                            _mm512_min_pd(m1, m3)));
}

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D) {
  if (LEN_1D <= 1) {
    if (LEN_1D == 1) result[0] = a[0];
    return;
  }
  double x = a[0];
  const int64_t M = LEN_1D - 1;
  const int64_t nch = M / 32;
  {
    const double *p = a + 1 + 32 * nch;
    const int64_t tail = M - 32 * nch;
    for (int64_t t = 0; t < tail; ++t)
      if (p[t] < x) x = p[t];
  }
  if (nch < 1024) {
    for (int64_t c = 0; c < nch; ++c) {
      double h = vmin32(a + 1 + 32 * c, x);
      if (h < x) x = h;
    }
  } else {
#pragma omp parallel for schedule(static) reduction(min:x)
    for (int64_t c = 0; c < nch; ++c) {
      double h = vmin32(a + 1 + 32 * c, x);
      if (h < x) x = h;
    }
  }
  result[0] = x;
}

int main(void) {
  printf("max_threads=%d\n", omp_get_max_threads());
  int64_t sizes[] = {100000000LL, 500000000LL, 1000000000LL, 1600000000LL};
  for (int s = 0; s < 4; ++s) {
    int64_t N = sizes[s];
    double *a = aligned_alloc(256, (size_t)N * 8);
    double r = 0;
    tsvc_2_s316_fp64(a, &r, N); /* warm */
    double best = 1e30;
    for (int rep = 0; rep < 2; ++rep) {
      double t0 = now_s();
      tsvc_2_s316_fp64(a, &r, N);
      double t1 = now_s();
      if (t1 - t0 < best) best = t1 - t0;
    }
    printf("N=%6.1fGB t=%.2fms agg=%.0fGB/s\n", N*8e-9, best*1e3, N*8.0/best/1e9);
    fflush(stdout);
    free(a);
  }
  return 0;
}
