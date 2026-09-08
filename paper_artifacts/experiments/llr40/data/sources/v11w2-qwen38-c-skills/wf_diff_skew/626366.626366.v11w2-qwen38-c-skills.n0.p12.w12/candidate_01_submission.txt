#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

/* NT_SCALE: nt = maxt * NT_SCALE / 100 */
#ifndef NT_SCALE
#define NT_SCALE 100
#endif
#ifndef PF_DEPTH
#define PF_DEPTH 4
#endif
#ifndef PF_HINT
#define PF_HINT _MM_HINT_T2
#endif

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n < 2) return;
  const int64_t ncols = n - 1;

  const int maxt = omp_get_max_threads();
  if (n < 1024 || maxt < 2) {
    for (int64_t i = 1; i < n; ++i) {
      double *restrict row = a + i * n;
      const double *restrict prev = a + (i - 1) * n;
      for (int64_t j = 0; j < ncols; ++j)
        row[j] = row[j] + prev[j] + prev[j + 1];
    }
    return;
  }

  int nt = (maxt * NT_SCALE) / 100;
  if (nt < 1) nt = 1;

  const int64_t nch = (ncols + 15) / 16;  /* 16-col chunks -> 128B-aligned boundaries */
  if (nt > nch) nt = (int)nch;
  const int64_t cbase = nch / nt, crem = nch % nt;

  int64_t *flags = (int64_t *)malloc((size_t)nt * 16 * sizeof(int64_t));  /* 128B per slot */
  for (int t = 0; t < nt; ++t) flags[t * 16] = 0;

  #pragma omp parallel num_threads(nt)
  {
    const int64_t tid = omp_get_thread_num();
    const int64_t c0 = tid * cbase + (tid < crem ? tid : crem);
    const int64_t cc = cbase + (tid < crem ? 1 : 0);
    const int64_t j0 = c0 * 16;
    int64_t j1 = (c0 + cc) * 16;
    if (j1 > ncols) j1 = ncols;
    const int need_r = (c0 + cc < nch);
    const int need_l = (tid > 0);
#if PF_DEPTH >= 1
    const int64_t cnt = j1 - j0;
#endif

    for (int64_t i = 1; i < n; ++i) {
      if (i > 1 && need_r) {
        const int64_t need = i - 1;
        while (__atomic_load_n(&flags[(tid + 1) * 16], __ATOMIC_ACQUIRE) < need)
          _mm_pause();
      }
      double *restrict row = a + i * n;
      const double *restrict prev = a + (i - 1) * n;
#if PF_DEPTH >= 1
      if (i + PF_DEPTH < n) {
        for (int64_t d = 1; d <= PF_DEPTH; ++d) {
          const double *p = a + (i + d) * n + j0;
          for (int64_t k = 0; k < cnt; k += 8)
            _mm_prefetch((const char *)(p + k), PF_HINT);
        }
      }
#endif
      for (int64_t j = j0; j < j1; ++j)
        row[j] = row[j] + prev[j] + prev[j + 1];
      if (need_l)
        __atomic_store_n(&flags[tid * 16], i, __ATOMIC_RELEASE);
    }
  }
  free(flags);
}
