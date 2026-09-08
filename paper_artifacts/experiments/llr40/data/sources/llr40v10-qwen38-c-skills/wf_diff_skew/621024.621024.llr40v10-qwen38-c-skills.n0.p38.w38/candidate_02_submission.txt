#include <stdint.h>
#include <omp.h>
#include <stdatomic.h>

/* Difference-diagonal stencil:  a[i,j] = a[i,j] + a[i-1,j] + a[i-1,j+1]
 * Row-major. The stencil reads only the previous row, so each row is an
 * independent unit-stride vector update and the rows form a one-deep
 * serial chain.
 *
 * Parallelization: column strips. Thread t owns columns [j0_t, j1_t)
 * across ALL rows. Per row the strip is a contiguous unit-stride block
 * (vectorized). The only cross-thread read is the single boundary element
 * prev[j1_t] (the j+1 read at j = j1_t-1): it is the first column of
 * thread t+1's strip, or the never-written last column for the last thread.
 * A per-row point-to-point handoff (release/acquire on done[]) syncs it,
 * avoiding a global barrier per row (which is catastrophic with many rows). */

static void wf_ser(double *restrict a, const int64_t L) {
  for (int64_t i = 1; i < L; ++i) {
    double *restrict row  = a + i * L;
    const double *restrict prev = a + (i - 1) * L;
    for (int64_t j = 0; j < L - 1; ++j)
      row[j] = row[j] + prev[j] + prev[j + 1];
  }
}

static _Alignas(64) _Atomic int64_t wf_done[1024];

static void wf_pt2p(double *restrict a, const int64_t L, int K) {
  const int64_t ncol = L - 1;
  for (int t = 0; t < K; ++t) wf_done[t] = 0;
  #pragma omp parallel num_threads(K)
  {
    const int t = omp_get_thread_num();
    const int64_t j0 = (int64_t)t * ncol / K;
    const int64_t j1 = (int64_t)(t + 1) * ncol / K;
    for (int64_t i = 1; i < L; ++i) {
      if (t < K - 1) {
        const int64_t need = i - 1;
        while (atomic_load_explicit(&wf_done[t + 1], memory_order_acquire) < need)
          __builtin_ia32_pause();
      }
      double *restrict row  = a + i * L;
      const double *restrict prev = a + (i - 1) * L;
      for (int64_t j = j0; j < j1; ++j)
        row[j] = row[j] + prev[j] + prev[j + 1];
      atomic_store_explicit(&wf_done[t], i, memory_order_release);
    }
  }
}

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t L = LEN_2D;
  if (L < 2) return;
  if (L < 256) {            /* small: team spawn + per-row handoff not worth it */
    wf_ser(a, L);
    return;
  }
  int K = omp_get_max_threads();
  if (K < 2) K = 2;
  if (K > 1024) K = 1024;
  wf_pt2p(a, L, K);
}
