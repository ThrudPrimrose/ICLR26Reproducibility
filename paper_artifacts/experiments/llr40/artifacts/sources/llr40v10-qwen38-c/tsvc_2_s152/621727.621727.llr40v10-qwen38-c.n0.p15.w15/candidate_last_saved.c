#include <omp.h>
#include <stdint.h>
#include <immintrin.h>

static void body_aligned(double *restrict a, double *restrict b, const double *restrict c,
                         const double *restrict d, const double *restrict e, const int64_t LEN_1D) {
  #pragma omp parallel for schedule(static)
  for (int64_t base = 0; base < LEN_1D; base += 8) {
    const int64_t rem = LEN_1D - base;
    if (rem >= 8) {
      __m512d bv = _mm512_mul_pd(_mm512_loadu_pd(d + base), _mm512_loadu_pd(e + base));
      _mm512_stream_pd(b + base, bv);
      __m512d av = _mm512_fmadd_pd(bv, _mm512_loadu_pd(c + base), _mm512_loadu_pd(a + base));
      _mm512_storeu_pd(a + base, av);
    } else {
      for (int64_t j = base; j < LEN_1D; ++j) {
        double t = d[j] * e[j];
        b[j] = t;
        a[j] += t * c[j];
      }
    }
  }
}

static void body_generic(double *restrict a, double *restrict b, const double *restrict c,
                         const double *restrict d, const double *restrict e, const int64_t LEN_1D) {
  #pragma omp parallel for schedule(static)
  for (int64_t base = 0; base < LEN_1D; base += 8) {
    const int64_t rem = LEN_1D - base;
    if (rem >= 8) {
      __m512d bv = _mm512_mul_pd(_mm512_loadu_pd(d + base), _mm512_loadu_pd(e + base));
      _mm512_storeu_pd(b + base, bv);
      __m512d av = _mm512_fmadd_pd(bv, _mm512_loadu_pd(c + base), _mm512_loadu_pd(a + base));
      _mm512_storeu_pd(a + base, av);
    } else {
      for (int64_t j = base; j < LEN_1D; ++j) {
        double t = d[j] * e[j];
        b[j] = t;
        a[j] += t * c[j];
      }
    }
  }
}

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e, const int64_t LEN_1D) {
  if ((uintptr_t)b & 63) {
    body_generic(a, b, c, d, e, LEN_1D);
  } else {
    body_aligned(a, b, c, d, e, LEN_1D);
  }
}
