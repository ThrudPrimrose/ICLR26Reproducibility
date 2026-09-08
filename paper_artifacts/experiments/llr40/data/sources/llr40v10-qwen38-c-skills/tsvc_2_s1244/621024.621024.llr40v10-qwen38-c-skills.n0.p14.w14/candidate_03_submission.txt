#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {

  const int64_t n = LEN_1D - 1;
  if (n <= 0) return;

  const int nt = ((uintptr_t)d & 63) == 0;

  #pragma omp parallel
  {
    const int64_t T = omp_get_num_threads();
    const int64_t tid = omp_get_thread_num();
    const int64_t base = n / T, rem = n % T;
    const int64_t s = tid * base + (tid < rem ? tid : rem);
    const int64_t e = s + base + (tid < rem ? 1 : 0);

    const double edge = a[e];
    #pragma omp barrier
    if (e > s) {
      int64_t i = s;
      /* peel to an 8-element boundary so NT stores are 64B-aligned */
      while (i + 16 <= e && (i & 7)) {
        const __m512d va = _mm512_loadu_pd(a + i + 1);
        const __m512d vb = _mm512_loadu_pd(b + i);
        const __m512d vc = _mm512_loadu_pd(c + i);
        const __m512d t = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(vb, _mm512_mul_pd(vc, vc)),
                                                      _mm512_mul_pd(vb, vb)), vc);
        _mm512_storeu_pd(a + i, t);
        _mm512_storeu_pd(d + i, _mm512_add_pd(t, va));
        i += 8;
      }
      if (nt) {
        for (; i + 32 <= e; i += 16) {
          __m512d va0 = _mm512_loadu_pd(a + i + 1);
          __m512d vb0 = _mm512_loadu_pd(b + i);
          __m512d vc0 = _mm512_loadu_pd(c + i);
          __m512d va1 = _mm512_loadu_pd(a + i + 9);
          __m512d vb1 = _mm512_loadu_pd(b + i + 8);
          __m512d vc1 = _mm512_loadu_pd(c + i + 8);
          __m512d t0 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(vb0, _mm512_mul_pd(vc0, vc0)),
                                                   _mm512_mul_pd(vb0, vb0)), vc0);
          __m512d t1 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(vb1, _mm512_mul_pd(vc1, vc1)),
                                                   _mm512_mul_pd(vb1, vb1)), vc1);
          _mm512_storeu_pd(a + i, t0);
          _mm512_storeu_pd(a + i + 8, t1);
          _mm512_stream_pd(d + i, _mm512_add_pd(t0, va0));
          _mm512_stream_pd(d + i + 8, _mm512_add_pd(t1, va1));
        }
      } else {
        for (; i + 32 <= e; i += 16) {
          __m512d va0 = _mm512_loadu_pd(a + i + 1);
          __m512d vb0 = _mm512_loadu_pd(b + i);
          __m512d vc0 = _mm512_loadu_pd(c + i);
          __m512d va1 = _mm512_loadu_pd(a + i + 9);
          __m512d vb1 = _mm512_loadu_pd(b + i + 8);
          __m512d vc1 = _mm512_loadu_pd(c + i + 8);
          __m512d t0 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(vb0, _mm512_mul_pd(vc0, vc0)),
                                                   _mm512_mul_pd(vb0, vb0)), vc0);
          __m512d t1 = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(vb1, _mm512_mul_pd(vc1, vc1)),
                                                   _mm512_mul_pd(vb1, vb1)), vc1);
          _mm512_storeu_pd(a + i, t0);
          _mm512_storeu_pd(a + i + 8, t1);
          _mm512_storeu_pd(d + i, _mm512_add_pd(t0, va0));
          _mm512_storeu_pd(d + i + 8, _mm512_add_pd(t1, va1));
        }
      }
      for (; i < e; i++) {
        const double t2 = (i + 1 == e) ? edge : a[i + 1];
        const double t = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
        a[i] = t;
        d[i] = t + t2;
      }
    }
  }
}
