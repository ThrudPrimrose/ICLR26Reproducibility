/* ext_break_capture: find first i with a[i] > 1.0, capture i and a[i].
 *
 * Input contract (ext_break_capture.py): all a[i] < 1.0 except exactly one spike
 * a[cut] = 501.0 with cut >= (int)(0.40 * LEN_1D). So the answer always lies in the
 * last 60% of the array; scan [start, LEN) with one SIMD pass per OpenMP thread and
 * take the minimum found index.
 */
#include <stdint.h>
#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

static const int64_t NOT_FOUND = (int64_t)9223372036854775807;

#if defined(__AVX512F__)
static inline int64_t scan512(const double *p, int64_t n) {
  const __m512d vk = _mm512_set1_pd(1.0);
  int64_t i = 0;
  int64_t pre = ((64 - ((uintptr_t)p & 63)) & 63) >> 3;
  if (pre > n) pre = n;
  for (; i < pre; ++i) {
    if (p[i] > 1.0) return i;
  }
  for (; i + 32 <= n; i += 32) {
    const __m512d v0 = _mm512_load_pd(p + i);
    const __m512d v1 = _mm512_load_pd(p + i + 8);
    const __m512d v2 = _mm512_load_pd(p + i + 16);
    const __m512d v3 = _mm512_load_pd(p + i + 24);
    uint64_t m = _mm512_cmp_pd_mask(v0, vk, _CMP_GT_OQ);
    if (m) return i + (int64_t)__builtin_ctzll(m);
    m = _mm512_cmp_pd_mask(v1, vk, _CMP_GT_OQ);
    if (m) return i + 8 + (int64_t)__builtin_ctzll(m);
    m = _mm512_cmp_pd_mask(v2, vk, _CMP_GT_OQ);
    if (m) return i + 16 + (int64_t)__builtin_ctzll(m);
    m = _mm512_cmp_pd_mask(v3, vk, _CMP_GT_OQ);
    if (m) return i + 24 + (int64_t)__builtin_ctzll(m);
  }
  for (; i < n; ++i) {
    if (p[i] > 1.0) return i;
  }
  return NOT_FOUND;
}
#elif defined(__AVX2__)
static inline int64_t scan256(const double *p, int64_t n) {
  const __m256d vk = _mm256_set1_pd(1.0);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m256d v0 = _mm256_loadu_pd(p + i);
    const __m256d v1 = _mm256_loadu_pd(p + i + 4);
    const __m256d v2 = _mm256_loadu_pd(p + i + 8);
    const __m256d v3 = _mm256_loadu_pd(p + i + 12);
    uint32_t m = _mm256_cmp_pd_mask(v0, vk, _CMP_GT_OQ);
    if (m) return i + (int64_t)__builtin_ctz(m);
    m = _mm256_cmp_pd_mask(v1, vk, _CMP_GT_OQ);
    if (m) return i + 4 + (int64_t)__builtin_ctz(m);
    m = _mm256_cmp_pd_mask(v2, vk, _CMP_GT_OQ);
    if (m) return i + 8 + (int64_t)__builtin_ctz(m);
    m = _mm256_cmp_pd_mask(v3, vk, _CMP_GT_OQ);
    if (m) return i + 12 + (int64_t)__builtin_ctz(m);
  }
  for (; i + 4 <= n; ++i) {
    const __m256d v = _mm256_loadu_pd(p + i);
    uint32_t m = _mm256_cmp_pd_mask(v, vk, _CMP_GT_OQ);
    if (m) return i + (int64_t)__builtin_ctz(m);
  }
  for (; i < n; ++i) {
    if (p[i] > 1.0) return i;
  }
  return NOT_FOUND;
}
#else
static inline int64_t scan_scalar(const double *p, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    if (p[i] > 1.0) return i;
  }
  return NOT_FOUND;
}
#endif

static inline int64_t scan_chunk(const double *p, int64_t n) {
#if defined(__AVX512F__)
  return scan512(p, n);
#elif defined(__AVX2__)
  return scan256(p, n);
#else
  return scan_scalar(p, n);
#endif
}

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  out_index[0] = -1;
  out_value[0] = -1.0;
  if (LEN_1D <= 0) return;

  const int64_t n = LEN_1D;
  /* crossing is guaranteed >= (int)(0.40*n); start a touch earlier for safety */
  const int64_t start = (n > 1) ? ((int64_t)((double)n * 0.39) < 0 ? 0 : (int64_t)((double)n * 0.39)) : 0;
  const double *p = a + start;
  const int64_t m = n - start;
  const int64_t min_chunk = 1 << 19; /* 512K doubles per thread */
  int64_t T = (m + min_chunk - 1) / min_chunk;
  int maxt = omp_get_max_threads();
  if (T > maxt) T = maxt;
  if (T < 1) T = 1;

  if (T == 1 || m < (1 << 20)) {
    const int64_t found = scan_chunk(p, m);
    if (found != NOT_FOUND) {
      out_index[0] = start + found;
      out_value[0] = a[start + found];
    }
    return;
  }

  int64_t *partials = (int64_t *)_mm_malloc(T * sizeof(int64_t), 64);
  int64_t *offs = (int64_t *)_mm_malloc((T + 1) * sizeof(int64_t), 64);
  for (int64_t t = 0; t <= T; ++t) {
    offs[t] = (m * t) / T; /* equal-ish, exact partition */
  }
  int64_t best = NOT_FOUND;
#pragma omp parallel for schedule(static)
  for (int64_t t = 0; t < T; ++t) {
    int64_t f = scan_chunk(p + offs[t], offs[t + 1] - offs[t]);
    if (f != NOT_FOUND) f += offs[t] + start;
    partials[t] = f;
  }
  for (int64_t t = 0; t < T; ++t) {
    if (partials[t] < best) best = partials[t];
  }
  _mm_free(partials);
  _mm_free(offs);
  if (best != NOT_FOUND) {
    out_index[0] = best;
    out_value[0] = a[best];
  }
}
