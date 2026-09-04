/* TSVC tsvc_2 s152 (fp64):  b[i] = d[i]*e[i];  a[i] += b[i]*c[i]
 *
 * AVX-512, 8 doubles per iteration. Non-temporal stores for a and b
 * (both are written once, never re-read) to keep the L1/L2 out of the
 * streaming write path. FMA for a[i]+t*c[i] -- bit-identical to the
 * reference compiled with -O3 -march=native (same contraction).
 * OpenMP-parallel over independent chunks. */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e, const int64_t LEN_1D) {
  const int64_t N8 = LEN_1D & ~7LL;
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < N8; i += 8) {
    const __m512d dv = _mm512_loadu_pd(d + i);
    const __m512d ev = _mm512_loadu_pd(e + i);
    const __m512d t  = _mm512_mul_pd(dv, ev);
    _mm512_stream_pd(b + i, t);
    const __m512d av = _mm512_loadu_pd(a + i);
    const __m512d cv = _mm512_loadu_pd(c + i);
    _mm512_stream_pd(a + i, _mm512_fmadd_pd(t, cv, av));
  }
  for (int64_t i = N8; i < LEN_1D; ++i) {
    const double t = d[i] * e[i];
    b[i] = t;
    a[i] += t * c[i];
  }
}
