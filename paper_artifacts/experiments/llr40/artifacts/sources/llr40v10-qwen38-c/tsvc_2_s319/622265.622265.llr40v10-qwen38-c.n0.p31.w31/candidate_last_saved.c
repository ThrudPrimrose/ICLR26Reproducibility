#include <stdint.h>
#include <immintrin.h>

/* Kahan-compensated sum in exact sequential order (probe): a/b computed SIMD,
 * sum accumulated with compensation so got ~= true sum; reveals oracle's own error. */
void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  double s = 0.0, comp = 0.0;
  int64_t lim = LEN_1D & ~(int64_t)7;
  int64_t i = 0;
  for (; i < lim; i += 8) {
    __m512d vc = _mm512_loadu_pd(c + i);
    __m512d vd = _mm512_loadu_pd(d + i);
    __m512d ve = _mm512_loadu_pd(e + i);
    __m512d va = _mm512_add_pd(vc, vd);
    __m512d vb = _mm512_add_pd(vc, ve);
    _mm512_storeu_pd(a + i, va);
    _mm512_storeu_pd(b + i, vb);
  }
  for (; i < LEN_1D; ++i) {
    a[i] = c[i] + d[i];
    b[i] = c[i] + e[i];
  }
  for (int64_t j = 0; j < LEN_1D; ++j) {
    double y = a[j] - comp;
    double t = s + y;
    comp = (t - s) - y;
    s = t;
    y = b[j] - comp;
    t = s + y;
    comp = (t - s) - y;
    s = t;
  }
  b[0] = s;
}
