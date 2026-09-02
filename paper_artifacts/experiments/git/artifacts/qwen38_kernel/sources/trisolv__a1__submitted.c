#include <stdint.h>
#include <immintrin.h>

static inline double hsum512(__m512d v){
    double d[8]; _mm512_storeu_pd(d,v);
    return ((d[0]+d[1])+(d[2]+d[3]))+((d[4]+d[5])+(d[6]+d[7]));
}

void trisolv_fp64(const double *restrict L, const double *restrict b, double *restrict x, int64_t N) {
    for (int64_t i = 0; i < N; ++i) {
        const double *Lrow = L + i*N;
        const int64_t n = i;
        const int64_t nstep = 32;              // 4 accumulators * 8
        const int64_t n4 = n & ~(nstep-1);
        __m512d a0=_mm512_setzero_pd(), a1=_mm512_setzero_pd(),
                a2=_mm512_setzero_pd(), a3=_mm512_setzero_pd();
        for (int64_t k=0; k<n4; k+=nstep) {
            __m512d l0=_mm512_loadu_pd(Lrow+k+0),  x0=_mm512_loadu_pd(x+k+0);
            __m512d l1=_mm512_loadu_pd(Lrow+k+8),  x1=_mm512_loadu_pd(x+k+8);
            __m512d l2=_mm512_loadu_pd(Lrow+k+16), x2=_mm512_loadu_pd(x+k+16);
            __m512d l3=_mm512_loadu_pd(Lrow+k+24), x3=_mm512_loadu_pd(x+k+24);
            a0=_mm512_fmadd_pd(l0,x0,a0);
            a1=_mm512_fmadd_pd(l1,x1,a1);
            a2=_mm512_fmadd_pd(l2,x2,a2);
            a3=_mm512_fmadd_pd(l3,x3,a3);
        }
        __m512d a = _mm512_add_pd(_mm512_add_pd(a0,a1),_mm512_add_pd(a2,a3));
        double s = hsum512(a);
        for (int64_t k=n4; k<n; ++k) s += Lrow[k]*x[k];
        x[i] = (b[i] - s) / Lrow[i];
    }
}
