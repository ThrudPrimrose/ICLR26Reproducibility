#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

/* Sliding-window AVX-512: each element of a is loaded once and reused for
 * up to 3 outputs in registers. 16 outputs per block:
 *   loads  = V0,V1 (128B) + 3 boundary scalars
 *   stores = 128B
 * => ~17.5 B/elem vs 40 B/elem for the 4-shifted-load form.
 * FP order matches the reference: ((x+y)+z) left to right. */

static inline __m512d rsh1v(__m512d V, double fill) {
  /* [fill, V0, V1, V2, V3, V4, V5, V6] */
  __m512d B = _mm512_set1_pd(fill);
  return _mm512_permutex2var_pd(V, _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 8), B);
}

static inline __m512d sh1v(__m512d V, double tail) {
  /* [V1, V2, V3, V4, V5, V6, V7, tail] */
  __m512d B = _mm512_set1_pd(tail);
  return _mm512_permutex2var_pd(V, _mm512_set_epi64(8, 7, 6, 5, 4, 3, 2, 1), B);
}

static inline __m512d sh2v(__m512d V, double t1, double t2) {
  /* [V2, V3, V4, V5, V6, V7, t1, t2] */
  __m512d B = _mm512_setr_pd(t1, t2, 0, 0, 0, 0, 0, 0);
  return _mm512_permutex2var_pd(V, _mm512_set_epi64(9, 8, 7, 6, 5, 4, 3, 2), B);
}

void fuse_stencil_through_transient_fp64(const double *restrict a,
                                         double *restrict out,
                                         const int64_t LEN_1D) {
  if (LEN_1D < 5) return;
  const int64_t i_last = LEN_1D - 3;          /* last valid i */
  const int64_t nb = (LEN_1D >= 18) ? ((LEN_1D - 17) / 16 + 1) : 0;
  const int64_t i_vec_end = 1 + 16 * nb;      /* first i left for scalar tail */

  const int64_t thr = (int64_t)omp_get_max_threads();
  if (nb >= (1 << 20) / 16 && thr >= 2) {
    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < nb; ++t) {
      const int64_t i0 = 1 + 16 * t;
      const __m512d V0 = _mm512_loadu_pd(a + i0);
      const __m512d V1 = _mm512_loadu_pd(a + i0 + 8);
      const double slo  = a[i0 - 1];
      const double s_h1 = a[i0 + 16];
      const double s_h2 = a[i0 + 17];
      const double v0_7 = a[i0 + 7];
      const double v1_0 = a[i0 + 8];
      const double v1_1 = a[i0 + 9];

      __m512d s1a = sh1v(V0, v1_0);
      __m512d m1 = _mm512_add_pd(_mm512_add_pd(rsh1v(V0, slo), V0), s1a);
      __m512d m2 = _mm512_add_pd(_mm512_add_pd(V0, s1a), sh2v(V0, v1_0, v1_1));
      _mm512_storeu_pd(out + i0, _mm512_mul_pd(m1, m2));

      __m512d s1b = sh1v(V1, s_h1);
      __m512d m3 = _mm512_add_pd(_mm512_add_pd(rsh1v(V1, v0_7), V1), s1b);
      __m512d m4 = _mm512_add_pd(_mm512_add_pd(V1, s1b), sh2v(V1, s_h1, s_h2));
      _mm512_storeu_pd(out + i0 + 8, _mm512_mul_pd(m3, m4));
    }
  } else {
    for (int64_t i = 1; i <= i_last; ++i)
      out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    return;
  }

  for (int64_t i = i_vec_end; i <= i_last; ++i)
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
}
