/* argmax_with_index -- SIMD (AVX-512/AVX2) + OpenMP rewrite of the TSVC running-max-with-index scan.
 *
 * One pass over `a`, split into contiguous per-thread spans. Each span scan runs four
 * independent 512-bit chains (elements 32k+c, c = 0..3) so four 64-byte loads are always in
 * flight; chain c keeps the running max of its own elements (broadcast) and the position of
 * the first occurrence of that max. At span end the four (max, first-position) pairs are
 * folded by (value desc, position asc), which is exactly the reference's first-occurrence
 * argmax. Strict `>` keeps first-occurrence tie-break; ordered comparisons never let NaN win,
 * matching the reference's `>` predicate (the one case where the reference itself returns
 * NaN, a[0] NaN, is an early exit). Spans are combined in address order with a strict `>`. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX512F__)

#define CV 8

static inline void scan_span(const double *restrict p, int64_t n, double *out_m, int64_t *out_i) {
  double m0 = -HUGE_VAL, m1 = -HUGE_VAL, m2 = -HUGE_VAL, m3 = -HUGE_VAL;
  int64_t i0 = -1, i1 = -1, i2 = -1, i3 = -1;
  int64_t i = 0;
  int64_t ng = n >> 5; /* 32 elements per group */
  for (; i < ng; ++i) {
    int64_t base = i << 5;
    __m512d v0 = _mm512_loadu_pd(p + base);
    __m512d v1 = _mm512_loadu_pd(p + base + 8);
    __m512d v2 = _mm512_loadu_pd(p + base + 16);
    __m512d v3 = _mm512_loadu_pd(p + base + 24);
    __mmask16 gt;
    gt = _mm512_cmp_pd_mask(v0, _mm512_set1_pd(m0), _CMP_GT_OQ);
    if (gt) {
      double vm = _mm512_reduce_max_pd(v0);
      i0 = base + __builtin_ctzll((unsigned long long)_mm512_cmp_pd_mask(v0, _mm512_set1_pd(vm), _CMP_EQ_OQ));
      m0 = vm;
    }
    gt = _mm512_cmp_pd_mask(v1, _mm512_set1_pd(m1), _CMP_GT_OQ);
    if (gt) {
      double vm = _mm512_reduce_max_pd(v1);
      i1 = base + 8 + __builtin_ctzll((unsigned long long)_mm512_cmp_pd_mask(v1, _mm512_set1_pd(vm), _CMP_EQ_OQ));
      m1 = vm;
    }
    gt = _mm512_cmp_pd_mask(v2, _mm512_set1_pd(m2), _CMP_GT_OQ);
    if (gt) {
      double vm = _mm512_reduce_max_pd(v2);
      i2 = base + 16 + __builtin_ctzll((unsigned long long)_mm512_cmp_pd_mask(v2, _mm512_set1_pd(vm), _CMP_EQ_OQ));
      m2 = vm;
    }
    gt = _mm512_cmp_pd_mask(v3, _mm512_set1_pd(m3), _CMP_GT_OQ);
    if (gt) {
      double vm = _mm512_reduce_max_pd(v3);
      i3 = base + 24 + __builtin_ctzll((unsigned long long)_mm512_cmp_pd_mask(v3, _mm512_set1_pd(vm), _CMP_EQ_OQ));
      m3 = vm;
    }
  }
  double best = -HUGE_VAL;
  int64_t bi = -1;
  double mv[4] = {m0, m1, m2, m3};
  int64_t iv[4] = {i0, i1, i2, i3};
  for (int c = 0; c < 4; ++c) {
    if (mv[c] > best) { best = mv[c]; bi = iv[c]; }
    else if (mv[c] == best && iv[c] >= 0 && (bi < 0 || iv[c] < bi)) bi = iv[c];
  }
  for (; i < n; ++i) {
    double v = p[i];
    if (v > best) { best = v; bi = i; }
  }
  *out_m = best;
  if (bi >= 0) *out_i = bi;
}

#elif defined(__AVX2__)

#define CV 4

static inline void scan_span(const double *restrict p, int64_t n, double *out_m, int64_t *out_i) {
  double m0 = -HUGE_VAL, m1 = -HUGE_VAL, m2 = -HUGE_VAL, m3 = -HUGE_VAL;
  int64_t i0 = -1, i1 = -1, i2 = -1, i3 = -1;
  int64_t i = 0;
  int64_t ng = n >> 4; /* 16 elements per group */
  for (; i < ng; ++i) {
    int64_t base = i << 4;
    __m256d v0 = _mm256_loadu_pd(p + base);
    __m256d v1 = _mm256_loadu_pd(p + base + 4);
    __m256d v2 = _mm256_loadu_pd(p + base + 8);
    __m256d v3 = _mm256_loadu_pd(p + base + 12);
    __mmask8 gt;
    gt = _mm256_cmp_pd_mask(v0, _mm256_set1_pd(m0), _CMP_GT_OQ);
    if (gt) {
      double vm = _mm256_reduce_max_pd(v0);
      i0 = base + __builtin_ctz((unsigned)_mm256_cmp_pd_mask(v0, _mm256_set1_pd(vm), _CMP_EQ_OQ));
      m0 = vm;
    }
    gt = _mm256_cmp_pd_mask(v1, _mm256_set1_pd(m1), _CMP_GT_OQ);
    if (gt) {
      double vm = _mm256_reduce_max_pd(v1);
      i1 = base + 4 + __builtin_ctz((unsigned)_mm256_cmp_pd_mask(v1, _mm256_set1_pd(vm), _CMP_EQ_OQ));
      m1 = vm;
    }
    gt = _mm256_cmp_pd_mask(v2, _mm256_set1_pd(m2), _CMP_GT_OQ);
    if (gt) {
      double vm = _mm256_reduce_max_pd(v2);
      i2 = base + 8 + __builtin_ctz((unsigned)_mm256_cmp_pd_mask(v2, _mm256_set1_pd(vm), _CMP_EQ_OQ));
      m2 = vm;
    }
    gt = _mm256_cmp_pd_mask(v3, _mm256_set1_pd(m3), _CMP_GT_OQ);
    if (gt) {
      double vm = _mm256_reduce_max_pd(v3);
      i3 = base + 12 + __builtin_ctz((unsigned)_mm256_cmp_pd_mask(v3, _mm256_set1_pd(vm), _CMP_EQ_OQ));
      m3 = vm;
    }
  }
  double best = -HUGE_VAL;
  int64_t bi = -1;
  double mv[4] = {m0, m1, m2, m3};
  int64_t iv[4] = {i0, i1, i2, i3};
  for (int c = 0; c < 4; ++c) {
    if (mv[c] > best) { best = mv[c]; bi = iv[c]; }
    else if (mv[c] == best && iv[c] >= 0 && (bi < 0 || iv[c] < bi)) bi = iv[c];
  }
  for (; i < n; ++i) {
    double v = p[i];
    if (v > best) { best = v; bi = i; }
  }
  *out_m = best;
  if (bi >= 0) *out_i = bi;
}

#else

static inline void scan_span(const double *restrict p, int64_t n, double *out_m, int64_t *out_i) {
  double m = -HUGE_VAL;
  int64_t idx = -1;
  for (int64_t i = 0; i < n; ++i) {
    double v = p[i];
    if (v > m) { m = v; idx = i; }
  }
  *out_m = m;
  if (idx >= 0) *out_i = idx;
}

#endif

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  if (LEN_1D == 1) { out_value[0] = a[0]; out_index[0] = 0; return; }
  if (a[0] != a[0]) { out_value[0] = a[0]; out_index[0] = 0; return; }

  int nt = omp_get_max_threads();
  if (nt < 2 || LEN_1D < (1 << 16)) {
    double m;
    int64_t idx = 0;
    scan_span(a, LEN_1D, &m, &idx);
    out_value[0] = m;
    out_index[0] = idx >= 0 ? idx : 0;
    return;
  }
  int64_t want = LEN_1D >> 13; /* ~8192 doubles per thread */
  if ((int64_t)nt > want) nt = (want > 1) ? (int)want : 1;

  int64_t per = LEN_1D / nt;
  int64_t extra = LEN_1D % nt;

  double *tmax = malloc((size_t)nt * sizeof(double));
  int64_t *tidx = malloc((size_t)nt * sizeof(int64_t));

  #pragma omp parallel num_threads(nt)
  {
    int t = omp_get_thread_num();
    int64_t lo = (int64_t)t * per + t;
    int64_t span = per + (t < extra);
    double m;
    int64_t idx = 0;
    scan_span(a + lo, span, &m, &idx);
    tmax[t] = m;
    tidx[t] = lo + (idx >= 0 ? idx : 0);
  }

  double best = tmax[0];
  int64_t bi = tidx[0];
  for (int t = 1; t < nt; ++t) {
    if (tmax[t] > best) { best = tmax[t]; bi = tidx[t]; }
  }
  out_value[0] = best;
  out_index[0] = bi;
  free(tmax);
  free(tidx);
}
