#include <stdint.h>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#define HAVE_AVX512 1
#endif

#if HAVE_AVX512
/* Plain multiply that the compiler cannot fold into an FMA: keeps each product
   rounded exactly like the scalar reference (no -ffp-contract is under our
   control, and it may contract intrinsic mul+add pairs at -O3 -march=native). */
static inline __m512d vmul_nofma(__m512d x, __m512d y) {
  __m512d r;
  __asm__ __volatile__("vmulpd %1, %2, %0" : "=y"(r) : "y"(x), "y"(y));
  return r;
}
#endif

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c,
                       double *restrict d, const int64_t LEN_1D) {
  const int64_t n = LEN_1D - 1;
  if (n <= 0) return;

#if HAVE_AVX512
  /* Single fused pass over [0, n):
       v[i]  = ((b[i] + c[i]*c[i]) + b[i]*b[i]) + c[i]
       a[i]  = v[i]
       d[i]  = v[i] + a_orig[i+1]
     a[i+1] is read before any store to a in each 16-wide block, and blocks
     of one thread are disjoint, so every a read sees the ORIGINAL value.
     Masked tail keeps the arithmetic path identical. */
  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (int64_t s = 0; s < n; s += 16) {
      const int64_t r = n - s;                 /* active lanes, 1..16 */
      const __mmask16 m = (r >= 16) ? (__mmask16)0xFFFFu : (__mmask16)((1u << r) - 1u);
      __m512d an0 = _mm512_maskz_loadu_pd(m, a + s + 1);
      __m512d an1 = _mm512_maskz_loadu_pd(m, a + s + 9);
      __m512d b0 = _mm512_maskz_loadu_pd(m, b + s);
      __m512d c0 = _mm512_maskz_loadu_pd(m, c + s);
      __m512d b1 = _mm512_maskz_loadu_pd(m, b + s + 8);
      __m512d c1 = _mm512_maskz_loadu_pd(m, c + s + 8);
      __m512d v0 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(b0, vmul_nofma(c0, c0)),
                                               vmul_nofma(b0, b0)),
                                 c0);
      __m512d v1 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(b1, vmul_nofma(c1, c1)),
                                               vmul_nofma(b1, b1)),
                                 c1);
      _mm512_mask_storeu_pd(d + s, m, _mm512_add_pd(v0, an0));
      _mm512_mask_storeu_pd(d + s + 8, m, _mm512_add_pd(v1, an1));
      _mm512_mask_storeu_pd(a + s, m, v0);
      _mm512_mask_storeu_pd(a + s + 8, m, v1);
    }
  }
#else
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) {
    d[i] = (b[i] + c[i] * c[i] + b[i] * b[i] + c[i]) + a[i + 1];
  }
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++) {
    a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
  }
#endif
}
