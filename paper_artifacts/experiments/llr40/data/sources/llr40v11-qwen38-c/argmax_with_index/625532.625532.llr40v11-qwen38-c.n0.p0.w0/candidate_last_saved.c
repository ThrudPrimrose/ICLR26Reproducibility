/* argmax_with_index: max value + first-occurrence index over a fp64 array.
 * OpenMP over contiguous chunks; AVX-512 one-pass per thread (AVX2 fallback).
 * Semantics identical to reference: strict '>', first occurrence wins ties. */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define HAVE_AVX512 1
#elif defined(__AVX2__)
#define HAVE_AVX512 0
#else
#define HAVE_AVX512 0
#define HAVE_SSE 1
#endif

#if HAVE_AVX512
static void chunk_argmax512(const double *restrict p, int64_t n, double *bv, int64_t *bi) {
  double best = p[0];
  int64_t bidx = 0;
  __m512d vb = _mm512_set1_pd(best);
  int64_t i = 0;
  int64_t nvec = n & ~7LL;
  for (; i < nvec; i += 8) {
    __m512d x = _mm512_loadu_pd(p + i);
    __mmask8 m = _mm512_cmp_pd_mask(x, vb, _CMP_GT_OQ);
    if (m) {
      double nm = _mm512_reduce_max_pd(x);
      __mmask8 eq = _mm512_cmp_pd_mask(x, _mm512_set1_pd(nm), _CMP_EQ_OQ);
      bidx = i + (int64_t)__builtin_ctzll((unsigned long long)eq);
      best = nm;
      vb = _mm512_set1_pd(best);
    }
  }
  for (; i < n; ++i) {
    double v = p[i];
    if (v > best) { best = v; bidx = i; }
  }
  *bv = best;
  *bi = bidx;
}
#elif HAVE_SSE
static void chunk_argmax512(const double *restrict p, int64_t n, double *bv, int64_t *bi) {
  double best = p[0];
  int64_t bidx = 0;
  __m256d vb = _mm256_set1_pd(best);
  int64_t i = 0;
  int64_t nvec = n & ~3LL;
  for (; i < nvec; i += 4) {
    __m256d x = _mm256_loadu_pd(p + i);
    __m256d m = _mm256_cmp_pd(x, vb, _CMP_GT_OQ);
    if (_mm256_movemask_pd(m)) {
      double t01[2], t23[2];
      _mm256_storeu_pd(t01, _mm256_max_pd(vb, x));
      _mm256_storeu_pd(t23, _mm256_max_pd(vb, _mm256_permute2f128_pd(x, x, 1)));
      double cand = t01[0];
      if (t01[1] > cand) cand = t01[1];
      if (t23[0] > cand) cand = t23[0];
      if (t23[1] > cand) cand = t23[1];
      __m256d eq = _mm256_cmp_pd(x, _mm256_set1_pd(cand), _CMP_EQ_OQ);
      int mm = _mm256_movemask_pd(eq);
      int lane = 0;
      while (!(mm & (1 << lane))) ++lane;
      bidx = i + lane;
      best = cand;
      vb = _mm256_set1_pd(best);
    }
  }
  for (; i < n; ++i) {
    double v = p[i];
    if (v > best) { best = v; bidx = i; }
  }
  *bv = best;
  *bi = bidx;
}
#endif

static double g_val[256];
static int64_t g_idx[256];

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n <= 0) { out_value[0] = 0.0; out_index[0] = 0; return; }
  if (n == 1) { out_value[0] = a[0]; out_index[0] = 0; return; }

#if HAVE_AVX512 || HAVE_SSE
  int nthreads = omp_get_max_threads();
  if (n < (int64_t)nthreads * 100000) {
    double bv; int64_t bi;
    chunk_argmax512(a, n, &bv, &bi);
    out_value[0] = bv;
    out_index[0] = bi;
    return;
  }
  int nt = 0;
  #pragma omp parallel
  {
    int t = omp_get_thread_num();
    int tot = omp_get_num_threads();
    nt = tot;
    int64_t per = (n + tot - 1) / tot;
    int64_t b = (int64_t)t * per;
    int64_t e = b + per;
    if (e > n) e = n;
    if (b < n) {
      double bv;
      int64_t bi;
      chunk_argmax512(a + b, e - b, &bv, &bi);
      g_val[t] = bv;
      g_idx[t] = b + bi;
    }
  }
  double best = g_val[0];
  int64_t bidx = g_idx[0];
  for (int t = 1; t < nt; ++t) {
    if (g_val[t] > best) {
      best = g_val[t];
      bidx = g_idx[t];
    } else if (g_val[t] == best && g_idx[t] < bidx) {
      bidx = g_idx[t];
    }
  }
  out_value[0] = best;
  out_index[0] = bidx;
#else
  double x = a[0];
  int64_t idx = 0;
  for (int64_t i = 1; i < n; ++i)
    if (a[i] > x) { x = a[i]; idx = i; }
  out_value[0] = x;
  out_index[0] = idx;
#endif
}
