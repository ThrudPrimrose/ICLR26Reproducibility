#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <immintrin.h>
#include <stdio.h>

typedef struct { double val; long long idx; } TRes;
#ifdef __AVX512F__
static inline double hmax512(__m512d v) {
    __m256d lo = _mm512_extractf64x4_pd(v, 0);
    __m256d hi = _mm512_extractf64x4_pd(v, 1);
    __m256d m = _mm256_max_pd(lo, hi);
    __m128d m2 = _mm_max_pd(_mm256_castpd256_pd128(m), _mm256_extractf128_pd(m, 1));
    __m128d m3 = _mm_max_pd(m2, _mm_unpackhi_pd(m2, m2));
    return _mm_cvtsd_f64(m3);
}
static inline int firstlane512(__m512d ad, double m) {
    __mmask8 mask = _mm512_cmp_pd_mask(ad, _mm512_set1_pd(m), _CMP_EQ_OQ);
    return __builtin_ctz((unsigned)mask);
}
static inline void scanvec(const double *restrict p, int64_t n, int64_t base, double *obv, long long *obi) {
    double bv = -INFINITY; long long bi = LLONG_MAX; int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512d x0=_mm512_loadu_pd(p+i), x1=_mm512_loadu_pd(p+i+8), x2=_mm512_loadu_pd(p+i+16), x3=_mm512_loadu_pd(p+i+24);
        __m512d a0=_mm512_abs_pd(x0),a1=_mm512_abs_pd(x1),a2=_mm512_abs_pd(x2),a3=_mm512_abs_pd(x3);
        double m0=hmax512(a0),m1=hmax512(a1),m2=hmax512(a2),m3=hmax512(a3);
        if(m0>bv){bv=m0;bi=base+i+firstlane512(a0,m0);}
        if(m1>bv){bv=m1;bi=base+i+8+firstlane512(a1,m1);}
        if(m2>bv){bv=m2;bi=base+i+16+firstlane512(a2,m2);}
        if(m3>bv){bv=m3;bi=base+i+24+firstlane512(a3,m3);}
    }
    for (; i + 8 <= n; i += 8) { __m512d ad=_mm512_abs_pd(_mm512_loadu_pd(p+i)); double m=hmax512(ad); if(m>bv){bv=m;bi=base+i+firstlane512(ad,m);} }
    for (; i < n; i++) { double v=fabs(p[i]); if(v>bv){bv=v;bi=base+i;} }
    *obv=bv; *obi=bi;
}
#else
static inline void scanvec(const double *restrict p, int64_t n, int64_t base, double *obv, long long *obi) {
    double bv=-INFINITY; long long bi=LLONG_MAX; for(int64_t i=0;i<n;i++){double v=fabs(p[i]); if(v>bv){bv=v;bi=base+i;}} *obv=bv;*obi=bi;
}
#endif

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D, const int64_t inc) {
    double t0 = omp_get_wtime();
    if (LEN_1D <= 0) { result[0] = 0.0; return; }
    const int nt = omp_get_max_threads();
    char *mem = (char *)malloc((size_t)nt * 128);
    TRes *tres = (TRes *)mem;
#pragma omp parallel
    {
        const int nt2 = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t chunk = LEN_1D / nt2, rem = LEN_1D % nt2;
        const int64_t lo = tid * chunk + (tid < rem ? tid : rem);
        const int64_t hi = lo + chunk + (tid < rem ? 1 : 0);
        double bv; long long bi;
        if (inc == 1) { scanvec(a + lo, hi - lo, lo, &bv, &bi); }
        else { bv=-INFINITY; bi=LLONG_MAX; for(int64_t i=lo;i<hi;i++){double v=fabs(a[i*inc]); if(v>bv){bv=v;bi=i;}} }
        tres[tid].val = bv; tres[tid].idx = bi;
#pragma omp barrier
        if (tid == 0) {
            double bestv=-INFINITY; long long besti=LLONG_MAX;
            for (int t=0;t<nt2;t++){ if(tres[t].val>bestv){bestv=tres[t].val;besti=tres[t].idx;} else if(tres[t].val==bestv && tres[t].idx<besti){besti=tres[t].idx;} }
            result[0] = bestv + (double)besti;
            double t1 = omp_get_wtime();
            printf("DIAG nt2=%d max_threads=%d self_time=%.4f ms LEN_1D=%lld inc=%lld\n", nt2, nt, (t1-t0)*1000.0, (long long)LEN_1D, (long long)inc); fflush(stdout);
        }
    }
    free(mem);
}
