#include <stdint.h>
#include <stdio.h>
#include <omp.h>

static inline unsigned long long rdtscv(void) {
  unsigned int lo, hi;
  __asm__ volatile ("rdtscp" : "=a"(lo), "=d"(hi) : : "rcx");
  return ((unsigned long long)hi << 32) | lo;
}

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int P = omp_get_num_threads();
    const int64_t per = (((N - 1) + P - 1) / P + 15) & ~(int64_t)15;
    double s = 0;
    unsigned long long t0 = rdtscv();
    for (int64_t j = 0; j < N; j++) {
      int64_t lo = j + 1 + (int64_t)tid * per;
      int64_t hi = lo + per; if (hi > N) hi = N;
      const double *rp = aa + j * N + lo;
      for (int64_t i = lo; i < hi; i++) s += rp[i - lo];
    }
    unsigned long long t1 = rdtscv();
    for (int64_t j = 0; j < N; j++) {
      int64_t lo = j + 1 + (int64_t)tid * per;
      int64_t hi = lo + per; if (hi > N) hi = N;
      const double *rp = aa + j * N + lo;
      int64_t cnt = hi - lo, k = 0;
      for (; k + 4 <= cnt; k += 4) s += rp[k] + rp[k+1] + rp[k+2] + rp[k+3];
      for (; k < cnt; k++) s += rp[k];
    }
    unsigned long long t2 = rdtscv();
    for (int64_t j = 0; j < N; j++) {
      int64_t lo = j + 1 + (int64_t)tid * per;
      int64_t hi = lo + per; if (hi > N) hi = N;
      const double *rp = aa + j * N + lo;
      int64_t cnt = hi - lo, k = 0;
      for (; k + 8 <= cnt; k += 8) {
        double x0=rp[k], x1=rp[k+1], x2=rp[k+2], x3=rp[k+3];
        double x4=rp[k+4], x5=rp[k+5], x6=rp[k+6], x7=rp[k+7];
        s += (x0+x1+x2+x3) + (x4+x5+x6+x7);
      }
      for (; k < cnt; k++) s += rp[k];
    }
    unsigned long long t3 = rdtscv();
    double aj = 0.5;
    for (int64_t j = 0; j < N; j++) {
      int64_t lo = j + 1 + (int64_t)tid * per;
      int64_t hi = lo + per; if (hi > N) hi = N;
      const double *rp = aa + j * N + lo;
      double *ap = a + lo;
      int64_t cnt = hi - lo, k = 0;
      for (; k + 4 <= cnt; k += 4) { ap[k]-=rp[k]*aj; ap[k+1]-=rp[k+1]*aj; ap[k+2]-=rp[k+2]*aj; ap[k+3]-=rp[k+3]*aj; }
      for (; k < cnt; k++) ap[k] -= rp[k]*aj;
    }
    unsigned long long t4 = rdtscv();
    #pragma omp barrier
    if (tid == 0) {
      double gh = 3.0;
      double gbs = (double)N * per * 8 / 1e9;
      double nsA = (t1-t0)/gh, nsB = (t2-t1)/gh, nsC = (t3-t2)/gh, nsD = (t4-t3)/gh;
      printf("P=%d per=%lld N=%ld\n", P, (long long)per, (long)N);
      printf("A read u1 : %8.0f ns  per-thread %6.2f GB/s\n", nsA, gbs*1e3/nsA);
      printf("B read u4 : %8.0f ns  per-thread %6.2f GB/s\n", nsB, gbs*1e3/nsB);
      printf("C read u8 : %8.0f ns  per-thread %6.2f GB/s\n", nsC, gbs*1e3/nsC);
      printf("D rmw u4  : %8.0f ns  per-thread %6.2f GB/s (3 streams)\n", nsD, gbs*3*1e3/nsD);
      printf("sink=%.3f\n", s);
      fflush(stdout);
    }
  }
}
