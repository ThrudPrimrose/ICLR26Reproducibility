#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#define NSLOT 128u
#define MAXT 48u
static _Alignas(64) uint32_t g_rowflags[MAXT][NSLOT];

static inline void band_update(const double *restrict prev, double *restrict cur,
                               int64_t lo, int64_t hi) {
  int64_t j = lo;
  for (; j + 7 < hi; j += 8) {
    __m512d vc = _mm512_loadu_pd(cur + j);
    __m512d v0 = _mm512_loadu_pd(prev + j);
    __m512d v1 = _mm512_loadu_pd(prev + j + 1);
    vc = _mm512_add_pd(vc, v0);
    vc = _mm512_add_pd(vc, v1);
    _mm512_storeu_pd(cur + j, vc);
  }
  for (; j < hi; ++j) {
    double t = cur[j] + prev[j];
    cur[j] = t + prev[j + 1];
  }
}

static inline void band_prefetch(const double *p, int64_t lo, int64_t hi) {
  for (int64_t j = lo; j < hi; j += 8)
    __builtin_prefetch(p + j, 0, 1); /* T2: pull into L2, not L1 */
}

void wf_diff_skew_fp64(double *restrict a, const int64_t N) {
  if (N < 2) return;
  const int64_t R = N - 1;
  int T = (int)omp_get_max_threads();
  if (T < 1) T = 1;
  if (T > (int)MAXT) T = (int)MAXT;
  if (T == 1 || N <= 1024) {
    for (int64_t i = 1; i < N; ++i)
      band_update(a + (i - 1) * N, a + i * N, 0, R);
    return;
  }
#pragma omp parallel num_threads(T)
  {
    const int t = omp_get_thread_num();
    for (unsigned s = 0; s < NSLOT; ++s) g_rowflags[t][s] = 0;
#pragma omp barrier
    const int64_t lo = (int64_t)t * R / T;
    const int64_t hi = (int64_t)(t + 1) * R / T;
    if (N > 2) band_prefetch(a + 2 * N, lo, hi);
    for (int64_t i = 1; i < N; ++i) {
      const int s = (int)((i - 1) & (NSLOT - 1));
      const uint32_t target = (uint32_t)((i - 1) >> 7) + 1u;
      const double *restrict prev = a + (i - 1) * N;
      double *restrict cur = a + i * N;
      if (i + 2 < N)
        band_prefetch(cur + 2 * N, lo, hi);
      band_update(prev, cur, lo, hi);
      __atomic_store_n(&g_rowflags[t][s], target, __ATOMIC_RELEASE);
      for (;;) {
        uint32_t missing = 0u;
        for (int t2 = 0; t2 < T; ++t2) {
          if (t2 == t) continue;
          if (__atomic_load_n(&g_rowflags[t2][s], __ATOMIC_ACQUIRE) < target) {
            missing = 1u;
            break;
          }
        }
        if (!missing) break;
        _mm_pause();
      }
    }
  }
}
