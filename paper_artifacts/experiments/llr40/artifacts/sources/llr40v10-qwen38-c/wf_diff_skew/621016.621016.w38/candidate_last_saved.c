/* wf_diff_skew: a[i,j] += a[i-1,j] + a[i-1,j+1]  (in place, row-major LEN_2D x LEN_2D)
 *
 * Rows form a serial dependency chain; within a row every j is independent.
 * Each OpenMP thread owns a contiguous column block [tW, (t+1)W) and walks the
 * rows serially inside ONE parallel region.  The only cross-thread dependence
 * of row r is a single element: the first column of the right neighbour's
 * block in row r-1, consumed by the LAST 8 columns of this thread's block.
 * The neighbour publishes a flag right after storing its first 8 columns of
 * the row, so the spin (just before that last vector) never idles a thread in
 * steady state.  Rows are processed with 512-bit vectors; the add order
 * matches the reference bit for bit: (cur + prev) + prev1.
 */
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

#if defined(__AVX512F__)
static inline void vec8(double *restrict cur, const double *restrict prev, int64_t j) {
  __m512d c = _mm512_loadu_pd(cur + j);
  __m512d p = _mm512_loadu_pd(prev + j);
  __m512d p1 = _mm512_loadu_pd(prev + j + 1);
  _mm512_storeu_pd(cur + j, _mm512_add_pd(_mm512_add_pd(c, p), p1));
}
#endif

/* block length is always >= 8 in the threaded path (see nt selection below) */
static void block_run(double *restrict cur, const double *restrict prev,
                      int64_t j0, int64_t j1, int64_t *flag, int64_t r,
                      const int64_t *neigh, int wait) {
  int64_t k;
#if defined(__AVX512F__)
  vec8(cur, prev, j0); /* first 8 columns: the only part other threads need */
  k = j0 + 8;
#else
  for (k = j0; k < j0 + 8 && k < j1; ++k) cur[k] = cur[k] + prev[k] + prev[k + 1];
#endif
  if (flag) __atomic_store_n(flag, r, __ATOMIC_RELEASE);
  const int64_t s = (k > j1 - 8) ? k : j1 - 8;
  for (; k + 8 <= s; k += 8) {
#if defined(__AVX512F__)
    vec8(cur, prev, k);
#else
    for (int64_t j = k; j < k + 8; ++j) cur[j] = cur[j] + prev[j] + prev[j + 1];
#endif
  }
  for (; k < s; ++k) cur[k] = cur[k] + prev[k] + prev[k + 1];
  if (wait)
    while (__atomic_load_n(neigh, __ATOMIC_ACQUIRE) < r - 1)
      __builtin_ia32_pause();
  if (j1 - s == 8) {
#if defined(__AVX512F__)
    vec8(cur, prev, s);
    return;
#endif
  }
  for (int64_t j = s + (j1 - s == 8 ? 8 : 0); j < j1; ++j)
    cur[j] = cur[j] + prev[j] + prev[j + 1];
}

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n < 2) return;
  const int64_t ncol = n - 1;

  int nt = omp_get_max_threads();
  if (nt < 1) nt = 1;
  /* keep the work per thread block large enough for threading to pay */
  while (nt > 1 && ncol / nt < 256) nt--;

  if (nt == 1) {
#if defined(__AVX512F__)
    for (int64_t r = 1; r < n; ++r) {
      double *cur = a + r * n;
      const double *prev = a + (r - 1) * n;
      for (int64_t j = 0; j + 8 <= ncol; j += 8) vec8(cur, prev, j);
      for (int64_t j = ncol & ~7LL; j < ncol; ++j) cur[j] = cur[j] + prev[j] + prev[j + 1];
    }
#else
    for (int64_t r = 1; r < n; ++r) {
      double *cur = a + r * n;
      const double *prev = a + (r - 1) * n;
      for (int64_t j = 0; j < ncol; ++j) cur[j] = cur[j] + prev[j] + prev[j + 1];
    }
#endif
    return;
  }

  const int64_t W = (ncol + nt - 1) / nt;
  int64_t *flags = (int64_t *)malloc((size_t)nt * sizeof(int64_t));
  for (int t = 0; t < nt; ++t) flags[t] = 0;

  omp_set_num_threads(nt);
  #pragma omp parallel
  {
    const int t = omp_get_thread_num();
    const int64_t j0 = (int64_t)t * W;
    const int64_t j1 = j0 + W < ncol ? j0 + W : ncol;
    const int hasR = (t + 1) * W < ncol; /* my last 8 cols need neighbour's first col */
    for (int64_t r = 1; r < n; ++r) {
      if (j0 < ncol) {
        block_run(a + r * n, a + (r - 1) * n, j0, j1,
                  &flags[t], r, &flags[t + 1], hasR && r > 1);
      } else {
        __atomic_store_n(&flags[t], r, __ATOMIC_RELEASE);
      }
    }
  }
  free(flags);
}
