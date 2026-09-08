#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <emmintrin.h>

#ifndef T_CAP
#define T_CAP 64
#endif

static int g_calls = 0;
static long long g_wait_ns = 0;
static long long g_spin = 0;

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 2) { if (N == 2) aa[3] = aa[0] + bb[3]; return; }
  const int64_t cols = N - 1;

  int T = (int)omp_get_max_threads();
  if (T > T_CAP) T = T_CAP;
  int64_t S = cols / 8;
  if (S > T) S = T;
  if (S < 1) S = 1;
  int64_t seg = (cols + S - 1) / S;
  seg = (seg + 7) & ~7LL;
  S = cols / seg;
  if (S < 1) S = 1;
  T = (int)S;

  if (N >= (1 << 20) || T <= 1) {
    for (int64_t i = 1; i < N; ++i) {
      double *restrict s = aa + (i - 1) * N;
      const double *restrict b = bb + i * N;
      double *restrict o = aa + i * N;
      for (int64_t j = 1; j < N; ++j) o[j] = s[j - 1] + b[j];
    }
    return;
  }

  static int gen = 0;
  const int64_t base = (int64_t)++gen << 20;
  static _Alignas(64) int64_t flag[T_CAP + 1][2];

  g_wait_ns = 0; g_spin = 0;
  const double t_start = omp_get_wtime();
  #pragma omp parallel num_threads(T)
  {
    const int64_t t = omp_get_thread_num();
    flag[t][0] = 0;
    flag[t][1] = 0;
    const int64_t j0 = 1 + t * seg;
    const int64_t je = (t == S - 1) ? N : j0 + seg;
    int64_t *ft = flag[t];
    const int64_t *fp = (t > 0) ? flag[t - 1] : NULL;
    double my_wait = 0.0;
    long long my_spin = 0;
    for (int64_t i = 1; i < N; ++i) {
      if (fp) {
        const int64_t need = base + i;
        double w0 = omp_get_wtime();
        while (__atomic_load_n(&fp[(i - 1) & 1], __ATOMIC_ACQUIRE) < need) {
          my_spin++;
          _mm_pause();
        }
        my_wait += omp_get_wtime() - w0;
      }
      double *restrict s = aa + (i - 1) * N;
      const double *restrict b = bb + i * N;
      double *restrict o = aa + i * N;
      for (int64_t j = j0; j < je; ++j) o[j] = s[j - 1] + b[j];
      __atomic_store_n(&ft[i & 1], base + i + 1, __ATOMIC_RELEASE);
    }
    __atomic_fetch_add(&g_wait_ns, (long long)(my_wait * 1e9), __ATOMIC_RELAXED);
    __atomic_fetch_add(&g_spin, my_spin, __ATOMIC_RELAXED);
  }
  long long tw, ts;
  const double t_end = omp_get_wtime();
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  tw = g_wait_ns; ts = g_spin;
  if (g_calls < 3) {
    printf("PROBE N=%lld max_threads=%d T_used=%d seg=%lld total=%.3fms sum_wait=%.3fms sum_spin=%lld\n",
           (long long)N, (int)omp_get_max_threads(), T, (long long)seg, (t_end - t_start) * 1e3,
           tw * 1e-6, ts);
    fflush(stdout);
    g_calls++;
  }
}
