#include <stdint.h>
#include <omp.h>

#ifdef __AVX512F__
#include <immintrin.h>
#define HAVE_AVX512 1
#endif

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
  #ifndef NT
  #define NT 24
  #endif
  omp_set_num_threads(NT);
  const int64_t n = LEN_1D;
  double total = 0.0;

#ifdef HAVE_AVX512
  const int64_t n8   = (n/8)*8;
  const int64_t nb   = n8/8;        /* number of 8-wide blocks   */
  const int64_t mainb= (nb/4)*4;    /* 8-blocks in parallel part */
  const int64_t nsup = mainb/4;     /* 32-element superblocks    */

  #pragma omp parallel reduction(+:total)
  {
    __m512d a0=_mm512_setzero_pd(), a1=_mm512_setzero_pd();
    __m512d a2=_mm512_setzero_pd(), a3=_mm512_setzero_pd();
    #pragma omp for schedule(static)
    for (int64_t s=0; s<nsup; s++) {
      const int64_t b = s*4;
      a0=_mm512_add_pd(a0,_mm512_loadu_pd(a+b*8));
      a1=_mm512_add_pd(a1,_mm512_loadu_pd(a+(b+1)*8));
      a2=_mm512_add_pd(a2,_mm512_loadu_pd(a+(b+2)*8));
      a3=_mm512_add_pd(a3,_mm512_loadu_pd(a+(b+3)*8));
    }
    double t[32];
    _mm512_storeu_pd(t+0,a0); _mm512_storeu_pd(t+8,a1);
    _mm512_storeu_pd(t+16,a2);_mm512_storeu_pd(t+24,a3);
    double l=0;
    for (int j=0;j<32;j++) l+=t[j];
    total += l;
  }
  /* tail: blocks mainb..nb-1 (0-3) + scalar remainder (n-n8 elems) */
  {
    __m512d a0=_mm512_setzero_pd(), a1=_mm512_setzero_pd();
    __m512d a2=_mm512_setzero_pd(), a3=_mm512_setzero_pd();
    for (int64_t b=mainb; b<nb; b++) {
      __m512d v=_mm512_loadu_pd(a+b*8);
      switch (b&3) {
        case 0: a0=_mm512_add_pd(a0,v); break;
        case 1: a1=_mm512_add_pd(a1,v); break;
        case 2: a2=_mm512_add_pd(a2,v); break;
        default:a3=_mm512_add_pd(a3,v); break;
      }
    }
    double t[32];
    _mm512_storeu_pd(t+0,a0); _mm512_storeu_pd(t+8,a1);
    _mm512_storeu_pd(t+16,a2);_mm512_storeu_pd(t+24,a3);
    double l=0;
    for (int j=0;j<32;j++) l+=t[j];
    for (int64_t i=n8;i<n;i++) l+=a[i];
    total += l;
  }
#else
  #pragma omp parallel reduction(+:total)
  {
    double l=0;
    #pragma omp for
    for (int64_t i=0;i<n;i++) l+=a[i];
    total += l;
  }
#endif
  sum_out[0] = total;
}
