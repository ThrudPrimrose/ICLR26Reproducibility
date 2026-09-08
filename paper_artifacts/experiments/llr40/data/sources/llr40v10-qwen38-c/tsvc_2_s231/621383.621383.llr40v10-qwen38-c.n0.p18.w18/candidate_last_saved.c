#include <stdint.h>
#include <omp.h>
#include <stdlib.h>
#include <immintrin.h>

/*
 * s231: aa[j][i] = aa[j-1][i] + bb[j][i]  (row-major, j=1..N-1, all i)
 * = per-column scan along rows: aa[j] = aa[0] + sum_{k=1..j} bb[k].
 *
 * Parallel (T = nthreads chunks), 3*N^2*8 bytes DRAM traffic:
 *   Pass A (parallel): chunk_sum[c] = sum of bb rows in chunk c, using 4
 *     independent accumulators (unrolled) to raise memory-level parallelism.
 *   Bases (sequential, tiny): baseval[c] = aa[0] + sum_{d<c} chunk_sum[d] (true base).
 *   Pass B (parallel): cur = baseval[c]; for j in chunk c: cur += bb[j]; aa[j] = cur.
 *     The aa write uses NON-TEMPORAL stores (written once, not re-read).
 * Only fp reassociation: the chunk-sum grouping in the bases (a few ulps).
 */
void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 1) return;

  int T = omp_get_max_threads();
  if (T < 1) T = 1;
  if (T > N - 1) T = (int)(N - 1);
  const int64_t base = (N - 1) / T;

  static double *csum = NULL, *baseval = NULL, *acc = NULL;
  static int64_t cached_N = 0, cached_T = 0;
  if (N != cached_N || T != cached_T) {
    free(csum); free(baseval); free(acc);
    csum    = (double *)malloc((size_t)T * (size_t)N * sizeof(double));
    baseval = (double *)malloc((size_t)T * (size_t)N * sizeof(double));
    acc     = (double *)malloc((size_t)T * 4 * (size_t)N * sizeof(double));
    cached_N = N; cached_T = T;
  }

  // Pass A: per-chunk row-sum of bb with 4 accumulators (parallel over chunks)
  #pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < T; ++c) {
    int64_t s = 1 + c * base;
    int64_t e = (c == T - 1) ? (N - 1) : (s + base - 1);
    double *a0 = acc + ((size_t)c * 4 + 0) * N;
    double *a1 = acc + ((size_t)c * 4 + 1) * N;
    double *a2 = acc + ((size_t)c * 4 + 2) * N;
    double *a3 = acc + ((size_t)c * 4 + 3) * N;
    for (int64_t i = 0; i < N; ++i) { a0[i] = 0.0; a1[i] = 0.0; a2[i] = 0.0; a3[i] = 0.0; }
    int64_t j = s;
    for (; j + 3 <= e; j += 4) {
      if (j + 6 < N) {
        __builtin_prefetch(bb + (j + 1) * N, 0, 1);
        __builtin_prefetch(bb + (j + 2) * N, 0, 1);
        __builtin_prefetch(bb + (j + 3) * N, 0, 1);
        __builtin_prefetch(bb + (j + 4) * N, 0, 1);
        __builtin_prefetch(bb + (j + 5) * N, 0, 1);
        __builtin_prefetch(bb + (j + 6) * N, 0, 1);
      }
      const double *b0 = bb + j * N;
      const double *b1 = bb + (j + 1) * N;
      const double *b2 = bb + (j + 2) * N;
      const double *b3 = bb + (j + 3) * N;
      for (int64_t i = 0; i < N; ++i) { a0[i] += b0[i]; a1[i] += b1[i]; a2[i] += b2[i]; a3[i] += b3[i]; }
    }
    while (j <= e) { const double *b = bb + j * N; for (int64_t i = 0; i < N; ++i) a0[i] += b[i]; ++j; }
    double *cs = csum + (size_t)c * N;
    for (int64_t i = 0; i < N; ++i) cs[i] = ((a0[i] + a1[i]) + (a2[i] + a3[i]));
  }

  // Bases: baseval[0] = aa[0]; baseval[c] = baseval[c-1] + csum[c-1]
  {
    double *b0 = baseval;
    const double *row0 = aa;
    for (int64_t i = 0; i < N; ++i) b0[i] = row0[i];
    for (int c = 1; c < T; ++c) {
      double *bc = baseval + (size_t)c * N;
      const double *bcp1 = baseval + (size_t)(c - 1) * N;
      const double *cs = csum + (size_t)(c - 1) * N;
      for (int64_t i = 0; i < N; ++i) bc[i] = bcp1[i] + cs[i];
    }
  }

  // Pass B: local scan from true base; aa written with non-temporal stores.
  #pragma omp parallel for schedule(static, 1)
  for (int c = 0; c < T; ++c) {
    int64_t s = 1 + c * base;
    int64_t e = (c == T - 1) ? (N - 1) : (s + base - 1);
    double *cur = baseval + (size_t)c * N;
    for (int64_t j = s; j <= e; ++j) {
      if (j + 6 < N) {
        __builtin_prefetch(bb + (j + 1) * N, 0, 1);
        __builtin_prefetch(bb + (j + 2) * N, 0, 1);
        __builtin_prefetch(bb + (j + 3) * N, 0, 1);
        __builtin_prefetch(bb + (j + 4) * N, 0, 1);
        __builtin_prefetch(bb + (j + 5) * N, 0, 1);
        __builtin_prefetch(bb + (j + 6) * N, 0, 1);
      }
      const double *b = bb + j * N;
      double *a = aa + j * N;
      size_t mis = (size_t)(uintptr_t)a & 63;
      int64_t i = 0;
      if (mis) {
        int64_t head = (64 - mis) / 8;
        if (head > N) head = N;
        for (; i < head; ++i) { double v = cur[i] + b[i]; cur[i] = v; a[i] = v; }
      }
      for (; i + 8 <= N; i += 8) {
        __m512d vc = _mm512_loadu_pd(cur + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        vc = _mm512_add_pd(vc, vb);
        _mm512_storeu_pd(cur + i, vc);
        _mm512_stream_pd(a + i, vc);
      }
      for (; i < N; ++i) { double v = cur[i] + b[i]; cur[i] = v; a[i] = v; }
    }
    _mm_sfence();
  }
  _mm_sfence();
}
