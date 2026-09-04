/* tsvc_2 s3110: global max of aa (LEN_2D x LEN_2D) with first row-major
 * occurrence (strict '>' tie-break of the reference);
 * bb[0] = maxv + (double)xindex + (double)yindex.
 *
 * Reference semantics, exactly replicated:
 *   - if aa[0][0] is NaN, every `v > maxv` is false, so the answer is NaN
 *     at (0,0)  ->  chksum = NaN.
 *   - otherwise maxv = max of all non-NaN elements (SSE 'max' and C '>' both
 *     ignore NaN), at its first row-major position.
 *
 * Single read pass. OpenMP threads own contiguous row blocks (deterministic
 * partition -> the ordered combine of per-thread partials reproduces the
 * serial first-occurrence tie-break). Inner scan: vectorized max with
 * first-occurrence tracking; ISA dispatched at runtime.
 * Blocks that achieve no max (empty or all-NaN) report pos = -1. A whole
 * array of -inf gives chksum = -inf regardless of the (0,-1) index, so the
 * sentinel is harmless. */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

typedef struct {
  double maxv;
  int64_t pos; /* flat row-major index, -1 = none */
} Part;

static Part scan_block(const double *a, int64_t n, int64_t r0, int64_t r1) {
  Part p;
  double maxv = -INFINITY;
  int64_t pos = -1;

  if (__builtin_cpu_supports("avx512f")) {
    __m512d vreg = _mm512_set1_pd(-INFINITY);
    for (int64_t i = r0; i < r1; ++i) {
      const double *row = a + i * n;
      int64_t j = 0;
      for (; j + 8 <= n; j += 8) {
        __m512d v = _mm512_loadu_pd(row + j);
        __m512d w = _mm512_max_pd(v, vreg);
        double lm = _mm512_reduce_max_pd(w);
        if (lm > maxv) {
          __mmask8 m2 = _mm512_cmp_pd_mask(v, _mm512_set1_pd(lm), _CMP_EQ_OQ);
          pos = i * n + j + (int64_t)__builtin_ctzll((unsigned long long)m2);
          maxv = lm;
          vreg = _mm512_set1_pd(lm);
        }
      }
      for (; j < n; ++j) {
        double v = row[j];
        if (v > maxv) { pos = i * n + j; maxv = v; vreg = _mm512_set1_pd(v); }
      }
    }
  } else if (__builtin_cpu_supports("avx2")) {
    __m256d vreg = _mm256_set1_pd(-INFINITY);
    for (int64_t i = r0; i < r1; ++i) {
      const double *row = a + i * n;
      int64_t j = 0;
      for (; j + 4 <= n; j += 4) {
        __m256d v = _mm256_loadu_pd(row + j);
        __m256d w = _mm256_max_pd(v, vreg);
        double t[4];
        _mm256_storeu_pd(t, w);
        double lm = t[0];
        if (t[1] > lm) lm = t[1];
        if (t[2] > lm) lm = t[2];
        if (t[3] > lm) lm = t[3];
        if (lm > maxv) {
          __m256d eq = _mm256_cmp_pd(v, _mm256_set1_pd(lm), _CMP_EQ_OQ);
          int m = _mm256_movemask_pd(eq);
          pos = i * n + j + (int64_t)__builtin_ctz((unsigned)m);
          maxv = lm;
          vreg = _mm256_set1_pd(lm);
        }
      }
      for (; j < n; ++j) {
        double v = row[j];
        if (v > maxv) { pos = i * n + j; maxv = v; vreg = _mm256_set1_pd(v); }
      }
    }
  } else {
    __m128d vreg = _mm_set1_pd(-INFINITY);
    for (int64_t i = r0; i < r1; ++i) {
      const double *row = a + i * n;
      int64_t j = 0;
      for (; j + 2 <= n; j += 2) {
        __m128d v = _mm_loadu_pd(row + j);
        __m128d w = _mm_max_pd(v, vreg);
        double t[2];
        _mm_storeu_pd(t, w);
        double lm = t[0] > t[1] ? t[0] : t[1];
        if (lm > maxv) {
          __m128d eq = _mm_cmpeq_pd(v, _mm_set1_pd(lm));
          int m = _mm_movemask_pd(eq) & 3;
          pos = i * n + j + (int64_t)__builtin_ctz((unsigned)m);
          maxv = lm;
          vreg = _mm_set1_pd(lm);
        }
      }
      for (; j < n; ++j) {
        double v = row[j];
        if (v > maxv) { pos = i * n + j; maxv = v; vreg = _mm_set1_pd(v); }
      }
    }
  }
  p.maxv = maxv; p.pos = pos;
  return p;
}

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n <= 0) { bb[0] = 0.0; return; }
  if (aa[0] != aa[0]) { bb[0] = NAN; return; } /* reference stays NaN at (0,0) */
  if (n == 1) { bb[0] = aa[0]; return; }

  int nt = omp_get_max_threads();
  if (nt < 1) nt = 1;
  Part *parts = (Part *)malloc(sizeof(Part) * (size_t)nt);

  #pragma omp parallel num_threads(nt)
  {
    int tid = omp_get_thread_num();
    int nt2 = omp_get_num_threads();
    int64_t r0 = (int64_t)tid * n / nt2;
    int64_t r1 = (int64_t)(tid + 1) * n / nt2;
    parts[tid] = scan_block(aa, n, r0, r1);
  }

  double maxv = -INFINITY;
  int64_t pos = -1;
  for (int t = 0; t < nt; ++t) {
    if (parts[t].pos >= 0 && (maxv == -INFINITY || parts[t].maxv > maxv)) {
      maxv = parts[t].maxv;
      pos = parts[t].pos;
    }
  }
  free(parts);

  int64_t xindex = pos / n;
  int64_t yindex = pos % n;
  double chksum = maxv + (double)(xindex) + (double)(yindex);
  bb[0] = chksum;
}
