#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
static inline int64_t count_pos(const double * __restrict__ p, int64_t n){
    int64_t i=0,cnt=0; const __m512d zero=_mm512_set1_pd(0.0);
    for(;i+8<=n;i+=8) cnt+=_mm_popcnt_u32((unsigned)_mm512_cmp_pd_mask(_mm512_loadu_pd(p+i),zero,_CMP_GT_OQ));
    for(;i<n;i++) cnt+=(p[i]>0.0);
    return cnt;
}
static inline void compact_pack(const double* __restrict__ src,const double* __restrict__ weight,
    double* __restrict__ packed,int64_t s,int64_t e,int64_t base){
    int64_t i=s,b=base; const __m512d zero=_mm512_set1_pd(0.0);
    for(;i+8<=e;i+=8){
        __m512d vs=_mm512_loadu_pd(src+i),vw=_mm512_loadu_pd(weight+i);
        __mmask8 m=_mm512_cmp_pd_mask(vs,zero,_CMP_GT_OQ);
        double tmp[8]; _mm512_storeu_pd(tmp,_mm512_mul_pd(vs,vw));
        unsigned um=(unsigned)m; while(um){int bit=__builtin_ctz(um);packed[b++]=tmp[bit];um&=um-1;}
    }
    for(;i<e;i++) if(src[i]>0.0) packed[b++]=src[i]*weight[i];
}
void compact_threshold_pack_fp64(int64_t *restrict out_count,double *restrict packed,
    const double *restrict src,const double *restrict weight,const int64_t LEN_1D,
    uint8_t *restrict workspace,int64_t workspace_size){
    (void)workspace;(void)workspace_size; int64_t N=LEN_1D;
    int nt=omp_get_max_threads(); if(nt<1)nt=1; if(nt>256)nt=256; if((int64_t)nt>N)nt=(int)N;
    int64_t counts[256],base[256];
    double p1sum=0,p3sum=0;
    for(int iter=0;iter<6;iter++){
        double t0=omp_get_wtime();
        #pragma omp parallel for schedule(static) num_threads(nt)
        for(int t=0;t<nt;t++){int64_t s=N*t/nt,e=N*(t+1)/nt;counts[t]=count_pos(src+s,e-s);}
        double t1=omp_get_wtime();
        int64_t run=0; for(int t=0;t<nt;t++){base[t]=run;run+=counts[t];} out_count[0]=run;
        #pragma omp parallel for schedule(static) num_threads(nt)
        for(int t=0;t<nt;t++){int64_t s=N*t/nt,e=N*(t+1)/nt;compact_pack(src,weight,packed,s,e,base[t]);}
        double t2=omp_get_wtime();
        if(iter>=2){p1sum+=(t1-t0);p3sum+=(t2-t1);}
    }
    p1sum/=4;p3sum/=4;
    printf("N=%lld nt=%d count=%lld p1=%.3f p3=%.3f total=%.3f p1bw=%.0f p3bw=%.0f\n",
      (long long)N,nt,(long long)out_count[0],p1sum*1e3,p3sum*1e3,(p1sum+p3sum)*1e3,
      (double)N*8/1e9/p1sum,(double)N*20/1e9/p3sum);
}
