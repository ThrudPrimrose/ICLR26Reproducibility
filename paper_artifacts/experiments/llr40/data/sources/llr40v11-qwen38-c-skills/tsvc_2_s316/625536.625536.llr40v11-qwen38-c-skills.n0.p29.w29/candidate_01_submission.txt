#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

/* TSVC s316: result[0] = min over a[0..LEN_1D-1].
 * Bandwidth-bound streaming min. Vectorized AVX-512 with exact reference
 * semantics (x = a[i] < x ? a[i] : x), so NaN/tie behavior matches bit-for-bit. */

static double chunk_min(const double *restrict p, int64_t len, double seed) {
  __m512d x0 = _mm512_set1_pd(seed), x1 = x0, x2 = x0, x3 = x0;
  int64_t k = 0;
  const int64_t n32 = len >> 5;           /* 32 doubles per unrolled step */
  for (; k < n32; ++k) {
    __m512d v0 = _mm512_loadu_pd(p + k*32);
    __m512d v1 = _mm512_loadu_pd(p + k*32 + 8);
    __m512d v2 = _mm512_loadu_pd(p + k*32 + 16);
    __m512d v3 = _mm512_loadu_pd(p + k*32 + 24);
    x0 = _mm512_mask_mov_pd(x0, _mm512_cmp_pd_mask(v0, x0, _CMP_LT_OQ), v0);
    x1 = _mm512_mask_mov_pd(x1, _mm512_cmp_pd_mask(v1, x1, _CMP_LT_OQ), v1);
    x2 = _mm512_mask_mov_pd(x2, _mm512_cmp_pd_mask(v2, x2, _CMP_LT_OQ), v2);
    x3 = _mm512_mask_mov_pd(x3, _mm512_cmp_pd_mask(v3, x3, _CMP_LT_OQ), v3);
  }
  k = n32 << 5;
  double m0[8], m1[8], m2[8], m3[8];
  _mm512_storeu_pd(m0, x0); _mm512_storeu_pd(m1, x1);
  _mm512_storeu_pd(m2, x2); _mm512_storeu_pd(m3, x3);
  double m = m0[0];
  for (int j = 1; j < 8; ++j) if (m0[j] < m) m = m0[j];
  for (int j = 0; j < 8; ++j) if (m1[j] < m) m = m1[j];
  for (int j = 0; j < 8; ++j) if (m2[j] < m) m = m2[j];
  for (int j = 0; j < 8; ++j) if (m3[j] < m) m = m3[j];
  for (; k < len; ++k) if (p[k] < m) m = p[k];
  return m;
}

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
  if (LEN_1D <= 1) { result[0] = a[0]; return; }
  const double seed = a[0];

  if (LEN_1D < (1 << 18)) {
    result[0] = chunk_min(a, LEN_1D, seed);
    return;
  }

  const int nt = omp_get_max_threads();
  double *part = (double *)malloc((size_t)nt * 8);
#pragma omp parallel num_threads(nt) shared(part)
  {
    const int t = omp_get_thread_num();
    const int ntt = omp_get_num_threads();
    int64_t chunk = (LEN_1D + ntt - 1) / ntt;
    int64_t beg = (int64_t)t * chunk;
    int64_t end = beg + chunk;
    if (end > LEN_1D) end = LEN_1D;
    part[t] = chunk_min(a + beg, end - beg, seed);
  }
  double x = seed;
  for (int t = 0; t < nt; ++t) if (part[t] < x) x = part[t];
  free(part);
  result[0] = x;
}
