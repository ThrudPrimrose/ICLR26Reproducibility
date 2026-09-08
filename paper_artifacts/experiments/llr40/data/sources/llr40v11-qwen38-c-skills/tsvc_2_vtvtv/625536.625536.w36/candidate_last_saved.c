#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
  const int64_t n32 = LEN_1D & ~31;
  #pragma omp parallel
  {
    const int64_t nt = omp_get_num_threads();
    const int64_t tid = omp_get_thread_num();
    const int64_t nb = n32 / 32;
    const int64_t lo = (nb * tid) / nt;
    const int64_t hi = (nb * (tid + 1)) / nt;
    for (int64_t bl = lo; bl < hi; ++bl) {
      const int64_t i = bl * 32;
      __m512d a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3;
      a0=_mm512_loadu_pd(a+i);    b0=_mm512_loadu_pd(b+i);
      a1=_mm512_loadu_pd(a+i+8);  b1=_mm512_loadu_pd(b+i+8);
      a2=_mm512_loadu_pd(a+i+16); b2=_mm512_loadu_pd(b+i+16);
      a3=_mm512_loadu_pd(a+i+24); b3=_mm512_loadu_pd(b+i+24);
      c0=_mm512_loadu_pd(c+i);    _mm512_storeu_pd(a+i,   _mm512_mul_pd(_mm512_mul_pd(a0,b0),c0));
      c1=_mm512_loadu_pd(c+i+8);  _mm512_storeu_pd(a+i+8, _mm512_mul_pd(_mm512_mul_pd(a1,b1),c1));
      c2=_mm512_loadu_pd(c+i+16); _mm512_storeu_pd(a+i+16,_mm512_mul_pd(_mm512_mul_pd(a2,b2),c2));
      c3=_mm512_loadu_pd(c+i+24); _mm512_storeu_pd(a+i+24,_mm512_mul_pd(_mm512_mul_pd(a3,b3),c3));
    }
    if (tid == 0)
      for (int64_t i = n32; i < LEN_1D; ++i)
        a[i] = a[i] * b[i] * c[i];
  }
}
