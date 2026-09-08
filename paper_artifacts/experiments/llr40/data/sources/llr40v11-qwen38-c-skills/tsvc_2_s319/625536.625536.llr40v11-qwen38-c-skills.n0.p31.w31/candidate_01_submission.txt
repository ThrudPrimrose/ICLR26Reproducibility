#include <stdint.h>
#include <stddef.h>
#include <omp.h>
#include <immintrin.h>

#define CHUNK 4096

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  double sum = 0.0;
  const int64_t nch = (LEN_1D + CHUNK - 1) / CHUNK;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (int64_t ch = 0; ch < nch; ++ch) {
    int64_t i0 = ch * (int64_t)CHUNK;
    int64_t i1 = i0 + (int64_t)CHUNK;
    if (i1 > LEN_1D) i1 = LEN_1D;
    double s = 0.0, t = 0.0;
    // Pass 1: a[i] = c[i] + d[i]
    {
      int64_t i = i0;
      while (i < i1 && (((uintptr_t)(const void *)a + 8 * (size_t)i) & 63u)) {
        double x = c[i] + d[i];
        a[i] = x;
        s += x;
        ++i;
      }
      __m512d sa = _mm512_setzero_pd();
      for (; i + 8 <= i1; i += 8) {
        __m512d vc = _mm512_loadu_pd(c + i);
        __m512d vd = _mm512_loadu_pd(d + i);
        __m512d va = _mm512_add_pd(vc, vd);
        _mm512_stream_pd(a + i, va);
        sa = _mm512_add_pd(sa, va);
      }
      double sa8[8];
      _mm512_storeu_pd(sa8, sa);
      s += ((sa8[0] + sa8[1]) + (sa8[2] + sa8[3])) + ((sa8[4] + sa8[5]) + (sa8[6] + sa8[7]));
      for (; i < i1; ++i) {
        double x = c[i] + d[i];
        a[i] = x;
        s += x;
      }
    }
    // Pass 2: b[i] = c[i] + e[i]
    {
      int64_t i = i0;
      while (i < i1 && (((uintptr_t)(const void *)b + 8 * (size_t)i) & 63u)) {
        double y = c[i] + e[i];
        b[i] = y;
        t += y;
        ++i;
      }
      __m512d sb = _mm512_setzero_pd();
      for (; i + 8 <= i1; i += 8) {
        __m512d vc = _mm512_loadu_pd(c + i);
        __m512d ve = _mm512_loadu_pd(e + i);
        __m512d vb = _mm512_add_pd(vc, ve);
        _mm512_stream_pd(b + i, vb);
        sb = _mm512_add_pd(sb, vb);
      }
      double sb8[8];
      _mm512_storeu_pd(sb8, sb);
      t += ((sb8[0] + sb8[1]) + (sb8[2] + sb8[3])) + ((sb8[4] + sb8[5]) + (sb8[6] + sb8[7]));
      for (; i < i1; ++i) {
        double y = c[i] + e[i];
        b[i] = y;
        t += y;
      }
    }
    _mm_sfence();
    sum += s + t;
  }
  b[0] = sum;
}
