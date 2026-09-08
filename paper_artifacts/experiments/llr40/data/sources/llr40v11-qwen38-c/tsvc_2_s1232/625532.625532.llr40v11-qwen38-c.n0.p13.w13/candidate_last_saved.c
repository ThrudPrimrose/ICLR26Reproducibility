#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
static inline long long now_ns(void){struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1000000000LL+ts.tv_nsec;}
static void rows_do(int64_t i0, int64_t i1, int64_t N, int64_t V, double *aa, const double *bb, const double *cc){
  for (int64_t i = i0; i < i1; ++i) {
    int64_t jmax = (V > 0) ? (i / V + 1) : N;
    if (jmax > N) jmax = N;
    double *a = aa + i * N;
    const double *b = bb + i * N;
    const double *c = cc + i * N;
    for (int64_t j = 0; j < jmax; ++j) a[j] = b[j] + c[j];
  }
}
static void run_dyn(int64_t N,int64_t V,double*aa,const double*bb,const double*cc){
#pragma omp parallel for schedule(dynamic,1)
  for (int64_t i=0;i<N;++i) {
    int64_t jmax = (V > 0) ? (i / V + 1) : N;
    if (jmax > N) jmax = N;
    double *a = aa + i * N;
    const double *b = bb + i * N;
    const double *c = cc + i * N;
    for (int64_t j = 0; j < jmax; ++j) a[j] = b[j] + c[j];
  }
}
static long long blo[4096], bhi[4096];
static void run_blocks(int64_t N,int64_t V,double*aa,const double*bb,const double*cc,int nb){
  static long long *pre=NULL; static int64_t pn=-1;
  if (pn < N) { pre = realloc(pre,(N+1)*8); pn=N; pre[0]=0; for(int64_t i=0;i<N;++i) pre[i+1]=pre[i]+((V>0)?(i/V+1):N); }
  long long target = pre[N]/nb; int64_t lo=0;
  for (int b=0;b<nb;++b){ blo[b]=lo; while(lo<N&&pre[lo]<target*(b+1))lo++; bhi[b]=lo; }
#pragma omp parallel for schedule(dynamic,1)
  for (int b=0;b<nb;++b) rows_do(blo[b],bhi[b],N,V,aa,bb,cc);
}
void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                       const int64_t LEN_2D, const int64_t VLEN) {
  const int64_t N = LEN_2D, V = VLEN;
  printf("N=%lld V=%lld\n",(long long)N,(long long)V);
  omp_set_num_threads(24);
  run_dyn(N,V,aa,bb,cc);
  const int nbs[8] = {48,96,96,192,192,384,768,192};
  long long t0,t1;
  for (int r=0;r<8;r++){
    t0=now_ns(); run_blocks(N,V,aa,bb,cc,nbs[r]); t1=now_ns();
    printf("  blk%d: %8.1f us\n", nbs[r], (t1-t0)/1000.0);
  }
  t0=now_ns(); run_dyn(N,V,aa,bb,cc); t1=now_ns();
  printf("  dyn  : %8.1f us\n", (t1-t0)/1000.0);
  omp_set_num_threads(omp_get_max_threads());
  fflush(stdout);
}
