#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#define T_SERIAL 256

/* Serial triangular update: for j<n, i in (j, n): a[i] -= aa[j*stride+i]*a[j] */
static void core_serial(double *restrict a, const double *restrict aa,
                        int64_t n, int64_t stride) {
  for (int64_t j = 0; j < n; j++) {
    const double aj = a[j];
    const double *restrict row = aa + j * stride;
    #pragma omp simd
    for (int64_t i = j + 1; i < n; i++) a[i] -= row[i] * aj;
  }
}

/* Cross-block update: for i2 in [0, n-m):
 *   a[m+i2] = fma(-aa[j*stride+m+i2], a[j], a[m+i2])  for j = 0..m-1, ascending.
 * Bit-identical to the reference's per-element FMA order (ascending j, one
 * rounding per update).  Each output COLUMN is an independent FMA chain, so
 * columns run in parallel (threads) and 8 columns per vector register (lanes);
 * the chain accumulator stays in a zmm across the whole j walk. */
static void col_chain_8(double *restrict col, const double *restrict ab,
                        const double *restrict ap, int64_t m, int64_t st) {
  __m512d v = _mm512_loadu_pd(col);
  int64_t j = 0;
  for (; j + 4 <= m; j += 4) {
    const __m512d r0 = _mm512_loadu_pd(ab + j * st);
    const __m512d r1 = _mm512_loadu_pd(ab + (j + 1) * st);
    const __m512d r2 = _mm512_loadu_pd(ab + (j + 2) * st);
    const __m512d r3 = _mm512_loadu_pd(ab + (j + 3) * st);
    v = _mm512_fnmadd_pd(r0, _mm512_set1_pd(ap[j]), v);
    v = _mm512_fnmadd_pd(r1, _mm512_set1_pd(ap[j + 1]), v);
    v = _mm512_fnmadd_pd(r2, _mm512_set1_pd(ap[j + 2]), v);
    v = _mm512_fnmadd_pd(r3, _mm512_set1_pd(ap[j + 3]), v);
    if (j + 72 <= m)
      _mm_prefetch((const char *)(ab + (j + 72) * st), _MM_HINT_T0);
  }
  for (; j < m; j++) {
    v = _mm512_fnmadd_pd(_mm512_loadu_pd(ab + j * st), _mm512_set1_pd(ap[j]), v);
  }
  _mm512_storeu_pd(col, v);
}

static void block_mv(double *restrict a, const double *restrict aa,
                     int64_t n, int64_t m, int64_t stride, int nt) {
  const int64_t len = n - m;
  if (nt > 1 && (int64_t)nt * len > 4096) {
    const int64_t full = len & ~7;
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int64_t i0 = 0; i0 < full; i0 += 8) {
      col_chain_8(a + m + i0, aa + m + i0, a, m, stride);
    }
    for (int64_t i2 = full; i2 < len; i2++) {
      double v = a[m + i2];
      for (int64_t j = 0; j < m; j++)
        v = fma(-aa[j * stride + m + i2], a[j], v);
      a[m + i2] = v;
    }
  } else {
    for (int64_t i2 = 0; i2 < len; i2++) {
      double v = a[m + i2];
      for (int64_t j = 0; j < m; j++)
        v = fma(-aa[j * stride + m + i2], a[j], v);
      a[m + i2] = v;
    }
  }
}

/* D&C: solve lower half, apply cross-block product, solve upper half.
 * The serial chain lives only on the critical path, O(log n) levels deep;
 * each level's cross product is a fully parallel matrix-vector product. */
static void rec(double *restrict a, const double *restrict aa,
                int64_t n, int64_t stride, int nt) {
  if (n <= T_SERIAL) { core_serial(a, aa, n, stride); return; }
  const int64_t m = n / 2;
  rec(a, aa, m, stride, nt);
  block_mv(a, aa, n, m, stride, nt);
  rec(a + m, aa + m * stride + m, n - m, stride, nt);
}

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
  if (LEN_2D <= T_SERIAL) { core_serial(a, aa, LEN_2D, LEN_2D); return; }
  const int nt = omp_get_max_threads();
  rec(a, aa, LEN_2D, LEN_2D, nt);
}
