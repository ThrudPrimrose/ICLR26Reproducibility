#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static double hsum512(__m512d v) {
  return (v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]);
}

static void run_fused(const double *restrict c, const double *restrict d,
                      double *restrict a, double *restrict b, int64_t n8, double *sum) {
  double p = 0.0;
  #pragma omp parallel for reduction(+:p) schedule(static)
  for (int64_t i = 0; i < n8; ++i) {
    __m512d cv = _mm512_loadu_pd(c + 8 * i);
    __m512d dv = _mm512_loadu_pd(d + 8 * i);
    __m512d rv = _mm512_add_pd(cv, dv);
    _mm512_stream_pd(a + 8 * i, rv);
    _mm512_stream_pd(b + 8 * i, rv);
    p += hsum512(_mm512_add_pd(rv, rv));
  }
  *sum += p;
}

static void run_general(const double *restrict c, const double *restrict d, const double *restrict e,
                        double *restrict a, double *restrict b, int64_t n8, double *sum) {
  double p = 0.0;
  #pragma omp parallel for reduction(+:p) schedule(static)
  for (int64_t i = 0; i < n8; ++i) {
    __m512d cv = _mm512_loadu_pd(c + 8 * i);
    __m512d dv = _mm512_loadu_pd(d + 8 * i);
    __m512d ev = _mm512_loadu_pd(e + 8 * i);
    __m512d av = _mm512_add_pd(cv, dv);
    __m512d bv = _mm512_add_pd(cv, ev);
    _mm512_stream_pd(a + 8 * i, av);
    _mm512_stream_pd(b + 8 * i, bv);
    p += hsum512(_mm512_add_pd(av, bv));
  }
  *sum += p;
}

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  double sum = 0.0;
  const int64_t n8 = LEN_1D / 8;

  if (e == d) {
    run_fused(c, d, a, b, n8, &sum);
    for (int64_t i = 8 * n8; i < LEN_1D; ++i) {
      a[i] = c[i] + d[i];
      b[i] = c[i] + d[i];
      sum += a[i] + b[i];
    }
  } else {
    run_general(c, d, e, a, b, n8, &sum);
    for (int64_t i = 8 * n8; i < LEN_1D; ++i) {
      a[i] = c[i] + d[i];
      b[i] = c[i] + e[i];
      sum += a[i] + b[i];
    }
  }
  _mm_sfence();
  b[0] = sum;
}
