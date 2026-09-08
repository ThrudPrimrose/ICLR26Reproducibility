/* argmax_with_index -- running maximum carrying value and first-occurrence index.
 *
 * Strategy:
 *  - Per thread: 16-lane (AVX-512) strided reduction. Each SIMD lane independently
 *    keeps the running max of its own subsequence (elements base+l, base+l+16, ...),
 *    so the inner loop has NO cross-lane dependency: load, compare, masked-blend.
 *  - First occurrence / ties: lane updates are "strictly greater", and the final
 *    merge of (value, index) pairs takes the higher value, and on exact tie the
 *    lower index.
 *  - NaN: a NaN is never strictly greater than anything, so it can never win; the
 *    only special case is a[0] == NaN, which the reference carries through forever.
 *  - Threads: contiguous static chunks, combined with the same tie-breaking merge.
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

typedef struct { double v; int64_t i; } amx_t;

/* Merge two (value, first-index) partial results: higher value wins, on exact
 * tie the lower index wins, NaN loses to anything. */
static inline amx_t amx_pick(amx_t a, amx_t b) {
  if (a.v != a.v) return b;            /* a is NaN  */
  if (b.v != b.v) return a;            /* b is NaN  */
  if (b.v > a.v) return b;
  if (a.v > b.v) return a;
  return (a.i <= b.i) ? a : b;         /* exact tie (incl. +-inf): first index wins */
}

/* ---------- AVX-512: 8 independent lane-wise running maxima (zmm = 8 doubles) ---------- */
__attribute__((target("avx512f")))
static amx_t amx_512(const double *restrict a, int64_t n, int64_t base) {
  __m512d xv = _mm512_set1_pd(-INFINITY);
  __m512i pos = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
  pos = _mm512_add_epi64(pos, _mm512_set1_epi64(base));
  __m512i idxv = pos;
  const __m512i step = _mm512_set1_epi64(8);

  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512d vA = _mm512_loadu_pd(a + i);
    __m512d vB = _mm512_loadu_pd(a + i + 8);
    __mmask8 mA = _mm512_cmp_pd_mask(vA, xv, _CMP_GT_OQ);
    idxv = _mm512_mask_blend_epi64(mA, idxv, pos);
    xv = _mm512_mask_mov_pd(xv, mA, vA);
    __mmask8 mB = _mm512_cmp_pd_mask(vB, xv, _CMP_GT_OQ);
    pos = _mm512_add_epi64(pos, step);
    idxv = _mm512_mask_blend_epi64(mB, idxv, pos);
    xv = _mm512_mask_mov_pd(xv, mB, vB);
    pos = _mm512_add_epi64(pos, step);
  }
  for (; i + 8 <= n; i += 8) {
    __m512d vA = _mm512_loadu_pd(a + i);
    __mmask8 mA = _mm512_cmp_pd_mask(vA, xv, _CMP_GT_OQ);
    idxv = _mm512_mask_blend_epi64(mA, idxv, pos);
    xv = _mm512_mask_mov_pd(xv, mA, vA);
    pos = _mm512_add_epi64(pos, step);
  }

  double dv[8];
  int64_t di[8];
  _mm512_store_pd(dv, xv);
  _mm512_store_si512(di, idxv);
  for (; i < n; ++i) {
    const double d = a[i];
    const int64_t gi = i + base;
    for (int l = 0; l < 8; ++l)
      if (d > dv[l]) { dv[l] = d; di[l] = gi; }
  }
  amx_t r = (amx_t){dv[0], di[0]};
  for (int l = 1; l < 8; ++l) r = amx_pick(r, (amx_t){dv[l], di[l]});
  return r;
}

/* ---------- AVX2 fallback: 4 lanes ---------- */
__attribute__((target("avx2")))
static amx_t amx_avx2(const double *restrict a, int64_t n, int64_t base) {
  __m256d xv = _mm256_set1_pd(-INFINITY);
  __m256i pos = _mm256_setr_epi64x(0, 1, 2, 3);
  pos = _mm256_add_epi64(pos, _mm256_set1_epi64x(base));
  __m256i idxv = pos;
  const __m256i step = _mm256_set1_epi64x(4);

  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256d vA = _mm256_loadu_pd(a + i);
    __m256d vB = _mm256_loadu_pd(a + i + 4);
    __m256d mA = _mm256_cmp_pd(vA, xv, _CMP_GT_OQ);
    idxv = _mm256_blendv_epi8(idxv, pos, _mm256_castpd_si256(mA));
    xv = _mm256_blendv_pd(xv, vA, mA);
    __m256d mB = _mm256_cmp_pd(vB, xv, _CMP_GT_OQ);
    pos = _mm256_add_epi64(pos, step);
    idxv = _mm256_blendv_epi8(idxv, pos, _mm256_castpd_si256(mB));
    xv = _mm256_blendv_pd(xv, vB, mB);
    pos = _mm256_add_epi64(pos, step);
  }
  for (; i + 4 <= n; i += 4) {
    __m256d vA = _mm256_loadu_pd(a + i);
    __m256d mA = _mm256_cmp_pd(vA, xv, _CMP_GT_OQ);
    idxv = _mm256_blendv_epi8(idxv, pos, _mm256_castpd_si256(mA));
    xv = _mm256_blendv_pd(xv, vA, mA);
    pos = _mm256_add_epi64(pos, step);
  }

  double dv[4];
  int64_t di[4];
  _mm256_store_pd(dv, xv);
  _mm256_store_si256((__m256i *)di, idxv);
  for (; i < n; ++i) {
    const double d = a[i];
    const int64_t gi = i + base;
    for (int l = 0; l < 4; ++l)
      if (d > dv[l]) { dv[l] = d; di[l] = gi; }
  }
  amx_t r = (amx_t){dv[0], di[0]};
  for (int l = 1; l < 4; ++l) r = amx_pick(r, (amx_t){dv[l], di[l]});
  return r;
}

/* ---------- scalar fallback ---------- */
static amx_t amx_scalar(const double *restrict a, int64_t n, int64_t base) {
  double x = -INFINITY;
  int64_t idx = -1;
  for (int64_t i = 0; i < n; ++i) {
    const double d = a[i];
    if (d > x) { x = d; idx = i + base; }
  }
  return (amx_t){x, idx};
}

static inline amx_t amx_one(const double *restrict a, int64_t n, int64_t base) {
  if (__builtin_cpu_supports("avx512f")) return amx_512(a, n, base);
  if (__builtin_cpu_supports("avx2"))    return amx_avx2(a, n, base);
  return amx_scalar(a, n, base);
}

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;
  if (a[0] != a[0]) { *out_value = a[0]; *out_index = 0; return; }

  int nt = (int)(LEN_1D >> 22);                 /* ~4 MiB of doubles per thread */
  if (nt < 1) nt = 1;
  int maxt = omp_get_max_threads();
  if (nt > maxt) nt = maxt;
  if (nt > 64)  nt = 64;

  if (nt == 1) {
    amx_t r = amx_one(a, LEN_1D, 0);
    *out_value = r.v;
    *out_index = r.i;
    return;
  }

  amx_t part[64];
#pragma omp parallel num_threads(nt)
  {
    const int64_t t = omp_get_thread_num();
    const int64_t T = nt;
    const int64_t lo = (LEN_1D * t) / T;
    const int64_t hi = (LEN_1D * (t + 1)) / T;
    part[t] = amx_one(a + lo, hi - lo, lo);
  }
  amx_t r = part[0];
  for (int t = 1; t < nt; ++t) r = amx_pick(r, part[t]);
  *out_value = r.v;
  *out_index = r.i;
}
