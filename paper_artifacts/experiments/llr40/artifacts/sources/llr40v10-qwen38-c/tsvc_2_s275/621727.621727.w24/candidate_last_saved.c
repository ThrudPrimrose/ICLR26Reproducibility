#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

#define NG 4            /* named groups (zmm accumulators) per unit */
#define UCOL (NG*8)     /* columns per unit = 32 */

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n < 2) return;
  const int ngroups = (int)((n + 7) / 8);
  unsigned char *cmask = (unsigned char*)malloc((size_t)ngroups);
  #pragma omp parallel for schedule(static)
  for (int g = 0; g < ngroups; g++) {
    unsigned char m = 0;
    const int base = g * 8;
    for (int k = 0; k < 8; k++)
      if (base + k < n && aa[base + k] > 0.0) m |= (unsigned char)(1u << k);
    cmask[g] = m;
  }

  const int nunits = (int)(n / UCOL);   /* full units only */
  #pragma omp parallel for schedule(static)
  for (int u = 0; u < nunits; u++) {
    const int i0 = u * UCOL;
    const int gb = i0 / 8;
    const __mmask8 m0 = cmask[gb+0], m1 = cmask[gb+1], m2 = cmask[gb+2], m3 = cmask[gb+3];
    double        *a0 = aa + i0 + 0,   *a1 = aa + i0 + 8,  *a2 = aa + i0 + 16, *a3 = aa + i0 + 24;
    const double  *b0 = bb + i0 + 0,   *b1 = bb + i0 + 8,  *b2 = bb + i0 + 16, *b3 = bb + i0 + 24;
    const double  *c0 = cc + i0 + 0,   *c1 = cc + i0 + 8,  *c2 = cc + i0 + 16, *c3 = cc + i0 + 24;
    __m512d x0 = _mm512_loadu_pd(a0);
    __m512d x1 = _mm512_loadu_pd(a1);
    __m512d x2 = _mm512_loadu_pd(a2);
    __m512d x3 = _mm512_loadu_pd(a3);
    int64_t off = n;                       /* row 1 */
    for (int64_t j = 1; j < n; j++) {
      x0 = _mm512_add_pd(x0, _mm512_mul_pd(_mm512_loadu_pd(b0+off), _mm512_loadu_pd(c0+off)));
      x1 = _mm512_add_pd(x1, _mm512_mul_pd(_mm512_loadu_pd(b1+off), _mm512_loadu_pd(c1+off)));
      x2 = _mm512_add_pd(x2, _mm512_mul_pd(_mm512_loadu_pd(b2+off), _mm512_loadu_pd(c2+off)));
      x3 = _mm512_add_pd(x3, _mm512_mul_pd(_mm512_loadu_pd(b3+off), _mm512_loadu_pd(c3+off)));
      _mm512_mask_store_pd(a0+off, m0, x0);
      _mm512_mask_store_pd(a1+off, m1, x1);
      _mm512_mask_store_pd(a2+off, m2, x2);
      _mm512_mask_store_pd(a3+off, m3, x3);
      off += n;
    }
  }

  const int64_t tail0 = (int64_t)nunits * UCOL;
  #pragma omp parallel for schedule(static)
  for (int64_t i = tail0; i < n; i++) {
    if (aa[i] > 0.0) {
      for (int64_t j = 1; j < n; j++)
        aa[j * n + i] = aa[(j - 1) * n + i] + bb[j * n + i] * cc[j * n + i];
    }
  }
  free(cmask);
}
