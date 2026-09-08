#include <stdint.h>
#include <omp.h>
#include <stdatomic.h>
#include <stdlib.h>
#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif
void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 1) return;
  const int64_t nj = N - 1;
  int T = omp_get_max_threads();
  if (T > 24) T = 24;
  if (T > nj) T = (int)nj;
  if (T < 1) T = 1;
  int64_t *bnd = (int64_t*)malloc(sizeof(int64_t)*(T+1));
  bnd[0] = 0;
  for (int t = 0; t < T-1; ++t) {
    int64_t rem  = nj - bnd[t];
    int64_t remt = T - t - 1;
    int64_t chunk = (rem + remt) / (remt + 1);
    int64_t achunk = (chunk / 8) * 8;
    if (achunk < 1) achunk = 1;
    if (bnd[t] + achunk > nj) achunk = nj - bnd[t];
    bnd[t+1] = bnd[t] + achunk;
  }
  bnd[T] = nj;
  _Atomic int *done = (_Atomic int*)malloc(sizeof(_Atomic int)*T);
  for (int t=0;t<T;++t) done[t] = 0;
  #pragma omp parallel num_threads(T)
  {
    const int tid = omp_get_thread_num();
    const int64_t lo = bnd[tid];
    const int64_t hi = bnd[tid+1];
    for (int64_t i = 1; i < N; ++i) {
      if (tid < T-1) {
        const int target = (int)(i-1);
        while (atomic_load_explicit(&done[tid+1], memory_order_acquire) < target) {
          __builtin_ia32_pause();
        }
      }
      double *restrict row  = a + i * N;
      const double *restrict prev = a + (i - 1) * N;
      for (int64_t j = lo; j < hi; ++j) {
        row[j] = row[j] + prev[j] + prev[j + 1];
      }
      atomic_store_explicit(&done[tid], (int)i, memory_order_release);
    }
  }
  free(done);
  free(bnd);
}
