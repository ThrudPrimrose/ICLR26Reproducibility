#include <stdint.h>
#include <stdio.h>
#include <omp.h>
#include <immintrin.h>

static void serial_fp64(double *restrict aa, const double *restrict bb, const int64_t N) {
  for (int64_t i = 1; i < N; ++i) {
    double *restrict a = aa + i * N;
    const double *restrict ap = aa + (i - 1) * N;
    const double *restrict b = bb + i * N;
    #pragma omp simd
    for (int64_t j = 1; j < N; ++j) a[j] = ap[j - 1] + b[j];
  }
}

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N < 2) return;

  const int maxT = omp_get_max_threads();
  if (maxT < 2 || (N - 1) / maxT < 16) {
    serial_fp64(aa, bb, N);
    return;
  }

  uint32_t flags[maxT];
  for (int i = 0; i < maxT; ++i) flags[i] = 0;
  uint64_t spin_cyc[maxT], tot_cyc[maxT]; int cpus[maxT];
  for (int i = 0; i < maxT; ++i) { spin_cyc[i] = tot_cyc[i] = 0; cpus[i] = -1; }

  #pragma omp parallel
  {
    const int t  = omp_get_thread_num();
    const int nt = omp_get_num_threads();

    const int64_t cnt  = N - 1;
    const int64_t base = cnt / nt;
    const int64_t rem  = cnt % nt;
    const int64_t c0   = 1 + (int64_t)t * base + (t < rem ? t : rem);
    const int64_t w    = base + (t < rem);

    unsigned a_, b_, c_, d_;
    __asm__ __volatile__ ("cpuid" : "=a"(a_), "=b"(b_), "=c"(c_), "=d"(d_) : "a"(1), "c"(0));
    cpus[t] = (int)((c_ >> 16) & 0xff);
    uint64_t spin = 0;
    const uint64_t t0 = __builtin_ia32_rdtsc();

    for (int64_t i = 1; i < N; ++i) {
      if (t > 0) {
        const uint64_t s0 = __builtin_ia32_rdtsc();
        while (__atomic_load_n(&flags[t - 1], __ATOMIC_ACQUIRE) < (uint32_t)(i - 1))
          __builtin_ia32_pause();
        spin += __builtin_ia32_rdtsc() - s0;
      }
      double *restrict a  = aa + i * N + c0;
      const double *restrict ap = aa + (i - 1) * N + c0 - 1;
      const double *restrict b  = bb + i * N + c0;

      for (int64_t j = 0; j < w; j += 8) {
        __builtin_prefetch(ap + j, 0, 1);
        __builtin_prefetch(b + j, 0, 0);
      }

      int64_t j = 0;
      while (j < w && ((uintptr_t)(a + j) & 63)) { a[j] = ap[j] + b[j]; ++j; }
      const int64_t tail = w > 0 ? w - 1 : 0;
      for (; j + 8 <= tail; j += 8)
        _mm512_stream_pd(a + j, _mm512_add_pd(_mm512_loadu_pd(ap + j), _mm512_loadu_pd(b + j)));
      for (; j < w; ++j) a[j] = ap[j] + b[j];

      __atomic_store_n(&flags[t], (uint32_t)i, __ATOMIC_RELEASE);

      if (i + 1 < N) {
        for (int64_t j = 0; j < w; j += 8)
          __builtin_prefetch(a + j, 0, 1);
      }
    }
    tot_cyc[t] = __builtin_ia32_rdtsc() - t0;
    spin_cyc[t] = spin;
  }

  printf("V2PROBE N=%ld\n", (long)N);
  for (int u = 0; u < maxT; ++u)
    if (cpus[u] >= 0)
      printf("  t%d cpu=%d tot=%.3fms spin=%.3fms spinpct=%.1f\n", u, cpus[u],
             tot_cyc[u] / 3e6, spin_cyc[u] / 3e6, 100.0 * (double)spin_cyc[u] / (double)tot_cyc[u]);
  fflush(stdout);
}
