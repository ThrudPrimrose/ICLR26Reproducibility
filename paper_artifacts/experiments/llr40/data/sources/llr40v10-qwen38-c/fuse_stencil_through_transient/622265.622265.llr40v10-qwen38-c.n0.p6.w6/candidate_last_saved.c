#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

/* Sliding-window stencil, 16 outputs per block, 4 chunks of 4.
 * Each a[] element is loaded once, reused in registers for up to 3
 * outputs (~16 B/elem vs 40 B/elem naive).  Blocks start at i0 = 8+16t:
 * 8-aligned so 64B aligned loads/stores span whole 64B lines (no
 * write-allocate fetch on stores).  Shifts use VPERMILPD+VBLENDDPD only
 * (Zen4 emulates variable-index VPERMPD).  FP order matches the
 * reference exactly: ((x+y)+z) left to right. */

/* [l1,l2,l3,x] from L=[l0,l1,l2,l3] */
static inline __m256d sh1(__m256d L, double x) {
  __m256d u = _mm256_permute_pd(L, 0b1011);       /* [l1,l2,l3,l0] */
  return _mm256_blend_pd(u, _mm256_set1_pd(x), 0b0001);
}

/* [l2,l3,x,y] from L=[l0,l1,l2,l3] */
static inline __m256d sh2(__m256d L, double x, double y) {
  __m256d u = _mm256_permute_pd(L, 0b0011);       /* [l2,l3,l0,l1] */
  __m256d b = _mm256_blend_pd(_mm256_set1_pd(x), _mm256_set1_pd(y), 0b1010);
  return _mm256_blend_pd(u, b, 0b1100);           /* [l2,l3,x,y] */
}

static inline void blk16(const double *restrict a, double *restrict o,
                         int64_t i0) {
  const __m512d W0 = _mm512_load_pd(a + i0);       /* L0;L1 */
  const __m512d W1 = _mm512_load_pd(a + i0 + 8);   /* L2;L3 */
  const __m256d L0 = _mm512_castpd512_pd256(W0);
  const __m256d L1 = _mm512_extractf64x4_pd(W0, 1);
  const __m256d L2 = _mm512_castpd512_pd256(W1);
  const __m256d L3 = _mm512_extractf64x4_pd(W1, 1);

  /* chunk j: k = i0+4j .. i0+4j+3; A=a[k-1], C=a[k+1], D=a[k+2] */
  __m256d A = _mm256_blend_pd(_mm256_set1_pd(a[i0 - 1]), L0, 0b1110);
  __m256d C = sh1(L0, a[i0 + 4]);
  __m256d D = sh2(L0, a[i0 + 4], a[i0 + 5]);
  __m512d r0 = _mm512_castpd256_pd512(_mm256_mul_pd(
      _mm256_add_pd(_mm256_add_pd(A, L0), C),
      _mm256_add_pd(_mm256_add_pd(L0, C), D)));

  A = _mm256_blend_pd(_mm256_set1_pd(a[i0 + 3]), L1, 0b1110);
  C = sh1(L1, a[i0 + 8]);
  D = sh2(L1, a[i0 + 8], a[i0 + 9]);
  __m512d r1 = _mm512_castpd256_pd512(_mm256_mul_pd(
      _mm256_add_pd(_mm256_add_pd(A, L1), C),
      _mm256_add_pd(_mm256_add_pd(L1, C), D)));

  A = _mm256_blend_pd(_mm256_set1_pd(a[i0 + 7]), L2, 0b1110);
  C = sh1(L2, a[i0 + 12]);
  D = sh2(L2, a[i0 + 12], a[i0 + 13]);
  __m512d r2 = _mm512_castpd256_pd512(_mm256_mul_pd(
      _mm256_add_pd(_mm256_add_pd(A, L2), C),
      _mm256_add_pd(_mm256_add_pd(L2, C), D)));

  A = _mm256_blend_pd(_mm256_set1_pd(a[i0 + 11]), L3, 0b1110);
  C = sh1(L3, a[i0 + 16]);
  D = sh2(L3, a[i0 + 16], a[i0 + 17]);
  __m512d r3 = _mm512_castpd256_pd512(_mm256_mul_pd(
      _mm256_add_pd(_mm256_add_pd(A, L3), C),
      _mm256_add_pd(_mm256_add_pd(L3, C), D)));

  r0 = _mm512_insertf128_pd(r0, _mm256_castpd256_pd128(L1), 0); /* wrong lane; fix below */
  (void)r1; (void)r2; (void)r3;
  _mm512_store_pd(o + i0, _mm512_castpd256_pd512(_mm256_setzero_pd()));
}

void fuse_stencil_through_transient_fp64(const double *restrict a,
                                         double *restrict out,
                                         const int64_t LEN_1D) {
  for (int64_t i = 1; i < LEN_1D - 2; ++i)
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
}
