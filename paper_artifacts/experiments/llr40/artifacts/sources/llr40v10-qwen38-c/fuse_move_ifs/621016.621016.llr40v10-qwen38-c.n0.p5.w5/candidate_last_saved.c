#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include <sys/syscall.h>

extern long syscall(long number, ...);

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void setmask(int lo, int hi, unsigned long *m) {
  memset(m, 0, sizeof(unsigned long[16]));
  for (int c = lo; c <= hi && c < 1024; c++) m[c >> 6] |= 1UL << (c & 63);
}

static void copy24(double *restrict b, const double *restrict src, int64_t N, int bind_lo, int bind_hi, int *ok) {
  #pragma omp parallel num_threads(24)
  {
    if (bind_lo >= 0) {
      unsigned long m[16];
      setmask(bind_lo, bind_hi, m);
      if (syscall(SYS_sched_setaffinity, 0, sizeof(m), (void *)m) != 0) {
        #pragma omp atomic
        *ok = *ok + 1;
      }
    }
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
      const double *si = src + i * N;
      double *bi = b + i * N;
      int64_t j = 0;
      for (; j + 16 <= N; j += 16) {
        __m512d v0 = _mm512_loadu_pd(si + j);
        __m512d v1 = _mm512_loadu_pd(si + j + 8);
        _mm512_storeu_pd(bi + j, v0);
        _mm512_storeu_pd(bi + j + 8, v1);
      }
      for (; j < N; ++j) bi[j] = si[j];
    }
  }
}

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  int ok1 = 0, ok2 = 0;
  double t0 = now_ms();
  copy24(b, src, N, 24, 47, &ok1);
  double t1 = now_ms();
  copy24(b, src, N, 0, 23, &ok2);
  double t2 = now_ms();

  const int do_b = (K > 0);
  #pragma omp parallel num_threads(24)
  {
    unsigned long m[16];
    setmask(24, 47, m);
    syscall(SYS_sched_setaffinity, 0, sizeof(m), (void *)m);
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
      const double *restrict si = src + i * N;
      if (cond[i] > 0.0) {
        double *restrict ai = a + i * N;
        if (do_b) {
          double *restrict bi = b + i * N;
          for (int64_t j = 0; j < N; ++j) { ai[j] = si[j] * 2.0; bi[j] = si[j] + 1.0; }
        } else {
          for (int64_t j = 0; j < N; ++j) ai[j] = si[j] * 2.0;
        }
      } else if (do_b) {
        double *restrict bi = b + i * N;
        for (int64_t j = 0; j < N; ++j) bi[j] = si[j] + 1.0;
      }
    }
  }
  double t3 = now_ms();
  printf("N=%lld node1=%.3f(ok%d) node0=%.3f(ok%d) final_bind1=%.3f\n",
         (long long)N, t1 - t0, ok1, t2 - t1, ok2, t3 - t2);
  fflush(stdout);
}
