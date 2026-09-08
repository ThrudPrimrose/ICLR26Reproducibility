#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
static volatile alignas(64) double VZERO[8] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static inline __m512d zero512(void){ return _mm512_loadu_pd((const double*)(void*)VZERO); }
static inline double hsum512(__m512d v){ alignas(64) double t[8]; _mm512_store_pd(t,v); return (t[0]+t[1]+t[2]+t[3])+(t[4]+t[5]+t[6]+t[7]); }
void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
  const int nt = omp_get_max_threads() > 0 ? omp_get_max_threads() : 1;
  double *partials = (double *)malloc(sizeof(double) * (size_t)nt);
  #pragma omp parallel num_threads(nt)
  {
    const int tid = omp_get_thread_num();
    const int64_t per = LEN_1D/nt, rem = LEN_1D%nt;
    const int64_t start = per*(int64_t)tid + ((int64_t)tid<rem?(int64_t)tid:rem);
    const int64_t len = per + ((int64_t)tid<rem?1:0);
    const double *p = a+start;
    const __m512d zero = zero512();
    __m512d a0=zero,a1=zero,a2=zero,a3=zero,a4=zero,a5=zero,a6=zero,a7=zero;
    int64_t i=0; const int64_t nvec=len-(len&63);
    for(;i<nvec;i+=64){ const double *q=p+i;
      if (i+128 < len) {
        _mm_prefetch((const char*)(p+i+128), _MM_HINT_T0);
        _mm_prefetch((const char*)(p+i+128+64), _MM_HINT_T0);
        _mm_prefetch((const char*)(p+i+128+128), _MM_HINT_T0);
      }
      __m512d v0=_mm512_loadu_pd(q),v1=_mm512_loadu_pd(q+8),v2=_mm512_loadu_pd(q+16),v3=_mm512_loadu_pd(q+24);
      __m512d v4=_mm512_loadu_pd(q+32),v5=_mm512_loadu_pd(q+40),v6=_mm512_loadu_pd(q+48),v7=_mm512_loadu_pd(q+56);
      a0=_mm512_add_pd(a0,_mm512_max_pd(v0,zero));a1=_mm512_add_pd(a1,_mm512_max_pd(v1,zero));
      a2=_mm512_add_pd(a2,_mm512_max_pd(v2,zero));a3=_mm512_add_pd(a3,_mm512_max_pd(v3,zero));
      a4=_mm512_add_pd(a4,_mm512_max_pd(v4,zero));a5=_mm512_add_pd(a5,_mm512_max_pd(v5,zero));
      a6=_mm512_add_pd(a6,_mm512_max_pd(v6,zero));a7=_mm512_add_pd(a7,_mm512_max_pd(v7,zero));
    }
    double local=0.0;
    local+=hsum512(a0)+hsum512(a1)+hsum512(a2)+hsum512(a3)+hsum512(a4)+hsum512(a5)+hsum512(a6)+hsum512(a7);
    for(;i<len;++i){double x=p[i];if(x>0.0)local+=x;}
    partials[tid]=local;
  }
  double sum=0.0; for(int t=0;t<nt;++t)sum+=partials[t]; free(partials); b[0]=sum;
}
