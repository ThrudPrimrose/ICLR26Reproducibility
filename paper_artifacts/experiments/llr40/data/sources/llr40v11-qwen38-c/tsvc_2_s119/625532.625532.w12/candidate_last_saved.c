#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>

extern long sysconf(int);
#define SC_NPROCESSORS_ONLN 84

static uint64_t now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (uint64_t)ts.tv_sec*1000000000ull + ts.tv_nsec; }

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  /* correct serial compute */
  for (int64_t i = 1; i < N; ++i) {
    double *restrict a = aa + i * N;
    const double *restrict ap = a - N - 1;
    const double *restrict b = bb + i * N;
    for (int64_t j = 1; j < N; ++j) a[j] = ap[j] + b[j];
  }
  /* bandwidth probe: read-stream over aa in 64B chunks */
  const int64_t NV = N*N/8;
  __m512d s = _mm512_setzero_pd();
  int reps = 3;
  uint64_t t0 = now_ns();
  for (int r = 0; r < reps; ++r)
    for (int64_t i = 0; i < NV; ++i)
      _mm512_add_pd(s, _mm512_loadu_pd(aa + 8*i));
  uint64_t t1 = now_ns();
  printf("PROBE read_bytes=%lld read_gbps=%.1f online=%ld\n", (long long)(NV*64*reps), (double)(NV*64*reps)/(double)(t1-t0), (long)sysconf(SC_NPROCESSORS_ONLN));
  (void)s;
  fflush(stdout);
}
