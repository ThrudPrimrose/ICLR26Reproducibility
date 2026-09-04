/* Optimized TSVC tsvc_2 s2710: elementwise with data-dependent branch.
 * No inter-iteration dependencies -> OpenMP over 8-chunks + masked SIMD.
 * FMA form of each expression matches the reference within last-ulp tolerance. */
#include <stdint.h>

#if defined(__AVX512F__)
#include <immintrin.h>

static void body_avx512(double *restrict a, double *restrict b, double *restrict c,
                        const double *restrict d, const double *restrict e,
                        int x0pos, int64_t i) {
  __m512d va = _mm512_loadu_pd(a + i);
  __m512d vb = _mm512_loadu_pd(b + i);
  __m512d vd = _mm512_loadu_pd(d + i);
  __m512d ve = _mm512_loadu_pd(e + i);
  __m512d vc = _mm512_loadu_pd(c + i);
  __mmask8 m = _mm512_cmp_pd_mask(va, vb, _CMP_GT_OQ);
  __m512d a_t = _mm512_fmadd_pd(vb, vd, va); /* a + b*d where m */
  __m512d b_f = _mm512_fmadd_pd(ve, ve, va); /* a + e*e where !m */
  __m512d c_t = _mm512_fmadd_pd(vd, vd, vc); /* c + d*d where m */
  __m512d c_f = x0pos ? _mm512_fmadd_pd(vd, vd, va) /* a + d*d */
                      : _mm512_fmadd_pd(ve, ve, vc); /* c + e*e */
  vc = _mm512_mask_blend_pd(m, c_f, c_t);
  _mm512_mask_storeu_pd(a + i, m, a_t);
  _mm512_mask_storeu_pd(b + i, ~m, b_f);
  _mm512_storeu_pd(c + i, vc);
}

#elif defined(__AVX2__)
#include <immintrin.h>

static void body_avx2(double *restrict a, double *restrict b, double *restrict c,
                      const double *restrict d, const double *restrict e,
                      int x0pos, int64_t i) {
  __m256d va = _mm256_loadu_pd(a + i);
  __m256d vb = _mm256_loadu_pd(b + i);
  __m256d vd = _mm256_loadu_pd(d + i);
  __m256d ve = _mm256_loadu_pd(e + i);
  __m256d vc = _mm256_loadu_pd(c + i);
  __m256d m = _mm256_cmp_pd(va, vb, _CMP_GT_OQ);
  __m256d a_t = _mm256_fmadd_pd(vb, vd, va);
  __m256d b_f = _mm256_fmadd_pd(ve, ve, va);
  __m256d c_t = _mm256_fmadd_pd(vd, vd, vc);
  __m256d c_f = x0pos ? _mm256_fmadd_pd(vd, vd, va) : _mm256_fmadd_pd(ve, ve, vc);
  vc = _mm256_blendv_pd(c_f, c_t, m);
  va = _mm256_blendv_pd(va, a_t, m);
  vb = _mm256_blendv_pd(b_f, vb, m);
  _mm256_storeu_pd(a + i, va);
  _mm256_storeu_pd(b + i, vb);
  _mm256_storeu_pd(c + i, vc);
}
#endif

static void body_scalar(double *restrict a, double *restrict b, double *restrict c,
                        const double *restrict d, const double *restrict e,
                        int x0pos, int64_t i, int64_t n) {
  for (; i < n; ++i) {
    if (a[i] > b[i]) {
      a[i] += b[i] * d[i];
      c[i] += d[i] * d[i];
    } else {
      b[i] = a[i] + e[i] * e[i];
      if (x0pos) c[i] = a[i] + d[i] * d[i];
      else c[i] += e[i] * e[i];
    }
  }
}

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n <= 0) return;

  if (n > 10) {
    const int x0pos = x[0] > 0.0;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 8) {
#if defined(__AVX512F__)
      if (i + 8 <= n)
        body_avx512(a, b, c, d, e, x0pos, i);
      else
        body_scalar(a, b, c, d, e, x0pos, i, n);
#elif defined(__AVX2__)
      if (i + 4 <= n)
        body_avx2(a, b, c, d, e, x0pos, i);
      else
        body_scalar(a, b, c, d, e, x0pos, i, n);
#else
      body_scalar(a, b, c, d, e, x0pos, i, n);
#endif
    }
  } else {
    /* LEN_1D <= 10: tiny scalar path; else branch keeps the x[0] test. */
    for (int64_t i = 0; i < n; ++i) {
      if (a[i] > b[i]) {
        a[i] += b[i] * d[i];
        c[i] = d[i] * e[i] + 1.0;
      } else {
        b[i] = a[i] + e[i] * e[i];
        if (x[0] > 0.0) c[i] = a[i] + d[i] * d[i];
        else c[i] += e[i] * e[i];
      }
    }
  }
}
