/* TSVC tsvc_2 kernel s316: x = a[0]; for i: if (a[i] < x) x = a[i]; result[0] = x;
 *
 * Min-reduction over LEN_1D doubles. Implemented as:
 *   - OpenMP: [1, LEN_1D) split into contiguous blocks, one per thread,
 *     combined with a min reduction (ordered '<' combine -> NaN-safe).
 *   - AVX-512 (8x fp64): per-lane ordered-masked min; a NaN can never win
 *     because the selection mask is an ordered quiet comparison.
 *   - Horizontal reduce via store + scalar ordered comparisons.
 * Bit-identical to the serial scan for all finite inputs; NaN propagation
 * matches the serial semantics (once x is NaN it stays NaN; NaN elements
 * are never adopted). Only the -0/+0 tie sign can vary across thread
 * split orderings (numerically identical, allclose-equal).
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline double hmin8(__m512d v) {
  double t[8];
  _mm512_store_pd(t, v);
  double r = t[0];
  if (t[1] < r) r = t[1];
  if (t[2] < r) r = t[2];
  if (t[3] < r) r = t[3];
  if (t[4] < r) r = t[4];
  if (t[5] < r) r = t[5];
  if (t[6] < r) r = t[6];
  if (t[7] < r) r = t[7];
  return r;
}

static inline double omind(double r, double v) { /* ordered, NaN-safe min(r,v) */
  if (v < r) r = v;
  return r;
}

/* min of p[0..n) into *x (which starts as a[0]) */
static void simd_block_min(const double *restrict p, int64_t n, double *x) {
  const double x0 = *x;
  __m512d a0 = _mm512_set1_pd(x0);
  __m512d a1 = _mm512_set1_pd(x0);
  __m512d a2 = _mm512_set1_pd(x0);
  __m512d a3 = _mm512_set1_pd(x0);
  const int64_t n8 = (n >> 3) << 3;
  int64_t i = 0;
  for (; i + 31 < n8; i += 32) {
    __m512d v;
    v = _mm512_loadu_pd(p + i);
    a0 = _mm512_mask_min_pd(a0, _mm512_cmp_pd_mask(v, a0, _CMP_LT_OQ), a0, v);
    v = _mm512_loadu_pd(p + i + 8);
    a1 = _mm512_mask_min_pd(a1, _mm512_cmp_pd_mask(v, a1, _CMP_LT_OQ), a1, v);
    v = _mm512_loadu_pd(p + i + 16);
    a2 = _mm512_mask_min_pd(a2, _mm512_cmp_pd_mask(v, a2, _CMP_LT_OQ), a2, v);
    v = _mm512_loadu_pd(p + i + 24);
    a3 = _mm512_mask_min_pd(a3, _mm512_cmp_pd_mask(v, a3, _CMP_LT_OQ), a3, v);
  }
  for (; i < n8; i += 8) {
    __m512d v = _mm512_loadu_pd(p + i);
    a0 = _mm512_mask_min_pd(a0, _mm512_cmp_pd_mask(v, a0, _CMP_LT_OQ), a0, v);
  }
  double r = x0; /* lane 0 of each a* already holds x0 first */
  r = omind(r, hmin8(a0));
  r = omind(r, hmin8(a1));
  r = omind(r, hmin8(a2));
  r = omind(r, hmin8(a3));
  for (; i < n; ++i)
    r = omind(r, p[i]);
  *x = r;
}

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D) {
  double x = a[0];
  const int64_t n = LEN_1D;
  long nt = 1;
  if (n >= 1048576)
    nt = omp_get_max_threads();
  if (nt > 1) {
    const int64_t m = n - 1;
    const int64_t base = m / nt, rem = m % nt;
    #pragma omp parallel for reduction(min: x) schedule(static)
    for (long t = 0; t < nt; ++t) {
      const int64_t start = 1 + t * base + (t < rem ? t : rem);
      const int64_t len = base + (t < rem ? 1 : 0);
      if (len > 0) simd_block_min(a + start, len, &x);
    }
  } else {
    if (n > 1) simd_block_min(a + 1, n - 1, &x);
  }
  result[0] = x;
}
