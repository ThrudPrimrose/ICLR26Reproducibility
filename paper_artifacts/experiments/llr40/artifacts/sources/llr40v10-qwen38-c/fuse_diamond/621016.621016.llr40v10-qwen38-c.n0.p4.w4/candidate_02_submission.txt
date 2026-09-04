#include <stdint.h>
#include <immintrin.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t n) {
  uintptr_t base = (uintptr_t)out;
  if (n >= 256 && (base & 63) == 0) {
    int64_t j = 0;
    int64_t n32 = n & ~(int64_t)31;
    for (; j < n32; j += 32) {
      __m512d x0 = _mm512_loadu_pd(a+j+0);
      __m512d x1 = _mm512_loadu_pd(a+j+8);
      __m512d x2 = _mm512_loadu_pd(a+j+16);
      __m512d x3 = _mm512_loadu_pd(a+j+24);
      __m512d t;
      t = _mm512_mul_pd(x0,x0);
      _mm512_stream_pd(out+j+0,  _mm512_mul_pd(_mm512_add_pd(t,_mm512_set1_pd(1.0)),_mm512_sub_pd(t,_mm512_set1_pd(1.0))));
      t = _mm512_mul_pd(x1,x1);
      _mm512_stream_pd(out+j+8,  _mm512_mul_pd(_mm512_add_pd(t,_mm512_set1_pd(1.0)),_mm512_sub_pd(t,_mm512_set1_pd(1.0))));
      t = _mm512_mul_pd(x2,x2);
      _mm512_stream_pd(out+j+16, _mm512_mul_pd(_mm512_add_pd(t,_mm512_set1_pd(1.0)),_mm512_sub_pd(t,_mm512_set1_pd(1.0))));
      t = _mm512_mul_pd(x3,x3);
      _mm512_stream_pd(out+j+24, _mm512_mul_pd(_mm512_add_pd(t,_mm512_set1_pd(1.0)),_mm512_sub_pd(t,_mm512_set1_pd(1.0))));
    }
    _mm_sfence();
    for (; j < n; ++j) { double t=a[j]*a[j]; out[j]=(t+1.0)*(t-1.0); }
  } else {
    for (int64_t i = 0; i < n; ++i) { double t=a[i]*a[i]; out[i]=(t+1.0)*(t-1.0); }
  }
}
