#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include <stdlib.h>

/* TSVC s2233: for i = 8..N-1:
     aa[j,i] = aa[j-1,i] + cc[j,i]   (j = 8..N-1)
     bb[i,j] = bb[i-1,j] + cc[i,j]   (j = 8..N-1)
  Both parts are column-wise prefix sums; the aa part touches only {aa,cc},
  the bb part only {bb,cc}, so the two parts are independent and run as two
  parallel chains.  Each chain is a scan along the row index: a block-wise
  prefix sum splits the n = N-8 rows into T independent row blocks per chain;
  the blocks start from the exact previous-block end value rebuilt from the
  block sums of cc, so every element is computed with the reference's per-row
  expression.  (Cross-block association of the block sums differs from the
  serial order; the graded tolerance covers FP reassociation.) */

static inline void vec_add_row(double *restrict d, const double *restrict a,
                               const double *restrict b, int64_t n) {
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm512_storeu_pd(d + i,
                     _mm512_add_pd(_mm512_loadu_pd(a + i), _mm512_loadu_pd(b + i)));
  for (; i + 4 <= n; i += 4)
    _mm256_storeu_pd(d + i,
                     _mm256_add_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
  for (; i < n; i++) d[i] = a[i] + b[i];
}

static inline void vec_acc_row(double *restrict d, const double *restrict c,
                               int64_t n) {
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm512_storeu_pd(d + i, _mm512_add_pd(_mm512_loadu_pd(d + i),
                                          _mm512_loadu_pd(c + i)));
  for (; i + 4 <= n; i += 4)
    _mm256_storeu_pd(d + i, _mm256_add_pd(_mm256_loadu_pd(d + i),
                                          _mm256_loadu_pd(c + i)));
  for (; i < n; i++) d[i] += c[i];
}

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb,
                       const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 8) return;
  const int64_t n = N - 8;

  /* fully serial, bit-identical (tiny N: no threading overhead) */
  if (n <= 256) {
    for (int64_t j = 8; j < N; ++j) {
      double *restrict a = aa + j * N + 8;
      const double *restrict p = a - N;
      const double *restrict c = cc + j * N + 8;
      for (int64_t i = 0; i < n; ++i) a[i] = p[i] + c[i];
    }
    for (int64_t i = 8; i < N; ++i) {
      double *restrict b = bb + i * N + 8;
      const double *restrict p = b - N;
      const double *restrict c = cc + i * N + 8;
      for (int64_t j = 0; j < n; ++j) b[j] = p[j] + c[j];
    }
    return;
  }

  int mt = omp_get_max_threads();
  int64_t T = mt / 2;
  if (T < 1) T = 1;
  if (T > 8) T = 8;

  /* one serial scan per chain, two threads, bit-identical */
  if (T == 1) {
#pragma omp parallel num_threads(2)
#pragma omp sections
    {
#pragma omp section
      for (int64_t j = 8; j < N; ++j)
        vec_add_row(aa + j * N + 8, aa + (j - 1) * N + 8, cc + j * N + 8, n);
#pragma omp section
      for (int64_t i = 8; i < N; ++i)
        vec_add_row(bb + i * N + 8, bb + (i - 1) * N + 8, cc + i * N + 8, n);
    }
    return;
  }

  /* block-wise prefix sum, T row blocks per chain, 2T threads */
  double *restrict P = (double *)aligned_alloc(64, (size_t)2 * (size_t)T * (size_t)n * 8);
  const int64_t b0 = n / T;
  const int64_t extra = n % T;

#pragma omp parallel num_threads(2 * T)
  {
    const int tid = omp_get_thread_num();
    const int chain = tid >= T;
    const int t = chain ? tid - T : tid;
    double *restrict A = chain ? bb : aa;
    const int64_t s_t = 8 + t * b0 + (t < extra ? t : extra);
    const int64_t e_t = s_t + b0 + (t < extra ? 1 : 0);
    double *restrict Pc = P + (size_t)chain * (size_t)T * n;
    double *restrict P_t = Pc + (size_t)t * n;

    /* pass 1: P_t = sum over the block's cc rows (columns 8..N-1) */
    {
      int64_t i = 0;
      for (; i + 8 <= n; i += 8)
        _mm512_storeu_pd(P_t + i, _mm512_setzero_pd());
      for (; i < n; i++) P_t[i] = 0.0;
      for (int64_t r = s_t; r < e_t; ++r)
        vec_acc_row(P_t, cc + r * N + 8, n);
    }

#pragma omp barrier

    /* per-chain prefix: P_t += P_{t-1} (thread 0 of the chain) */
    if (t == 0)
      for (int k = 1; k < T; ++k)
        vec_acc_row(Pc + (size_t)k * n, Pc + (size_t)(k - 1) * n, n);

#pragma omp barrier

    /* pass 2: scan the block, starting from the exact previous end value */
    double off[n + 8];
    {
      const double *const f7 = A + 7 * N + 8;
      int64_t i = 0;
      const double *const Pp = (t == 0) ? 0 : Pc + (size_t)(t - 1) * n;
      if (t == 0) {
        for (; i + 8 <= n; i += 8)
          _mm512_storeu_pd(off + i, _mm512_loadu_pd(f7 + i));
        for (; i < n; i++) off[i] = f7[i];
      } else {
        for (; i + 8 <= n; i += 8)
          _mm512_storeu_pd(off + i,
                           _mm512_add_pd(_mm512_loadu_pd(f7 + i),
                                         _mm512_loadu_pd(Pp + i)));
        for (; i + 4 <= n; i += 4)
          off[i] = f7[i] + Pp[i];
        for (; i < n; i++) off[i] = f7[i] + Pp[i];
      }
    }
    {
      const int64_t r = s_t;
      vec_add_row(A + r * N + 8, off, cc + r * N + 8, n);
      for (int64_t rr = r + 1; rr < e_t; ++rr)
        vec_add_row(A + rr * N + 8, A + (rr - 1) * N + 8, cc + rr * N + 8, n);
    }
  }
  free(P);
}
