#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  const int64_t N = LEN_1D - 1; /* iterations i = 0 .. N-1 */
  if (N <= 0) return;

  int nt = (int)omp_get_max_threads();
  if (nt < 1) nt = 1;
  if (nt > N) nt = (int)N;

  if (N < 131072 || nt < 2) {
    /* serial vector path (also avoids fork overhead on small inputs) */
    int64_t i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= N; i += 8) {
      __m512d av = _mm512_loadu_pd(a + i + 1); /* a[i+1 .. i+8]: original values, read before overwrite */
      __m512d bv = _mm512_loadu_pd(b + i);
      _mm512_storeu_pd(a + i, _mm512_add_pd(av, bv));
    }
#endif
    for (; i < N; ++i) a[i] = a[i + 1] + b[i];
    return;
  }

  /* Per-chunk save of the chunk-start element, so the previous chunk's
   * last iteration (which reads a[hi_prev+1] == a[lo_this]) can use the
   * original value instead of racing with this chunk's first write. */
  double *saved = (double *)malloc((size_t)nt * sizeof(double));

#pragma omp parallel num_threads(nt)
  {
    const int tid = omp_get_thread_num();
    const int64_t base = N / nt, rem = N % nt;
    const int64_t lo = (int64_t)tid * base + (tid < rem ? tid : rem);
    const int64_t hi = lo + base + (tid < rem ? 1 : 0) - 1;

    saved[tid] = a[lo]; /* phase 1 */
#pragma omp barrier
    /* phase 2: within-chunk reads a[i+1] happen before the same chunk
     * overwrites them, so only a[hi+1] needs the saved value. */
    int64_t i = lo;
#if defined(__AVX512F__)
    for (; i + 8 <= hi; i += 8) {
      __m512d av = _mm512_loadu_pd(a + i + 1);
      __m512d bv = _mm512_loadu_pd(b + i);
      _mm512_storeu_pd(a + i, _mm512_add_pd(av, bv));
    }
#endif
    for (; i < hi; ++i) a[i] = a[i + 1] + b[i];
    if (tid + 1 < nt) a[hi] = saved[tid + 1] + b[hi];
    else a[hi] = a[hi + 1] + b[hi];
  }

  free(saved);
}
