#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
static double now_ms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  const int N = (int)LEN_2D; // ~13796
  int nts[4]={2,4,8,16};
  for(int k=0;k<4;k++){
    int nt=nts[k];
    // measure N empty omp-for barriers
    #pragma omp parallel num_threads(nt)
    {
      double t0=now_ms();
      for(int i=0;i<N;i++){
        #pragma omp for
        for(int j=0;j<1;j++); // empty, just for the barrier
      }
      double t1=now_ms();
      if(omp_get_thread_num()==0) printf("barrier-only nt=%2d N=%d: total=%.1fms per-barrier=%.1f us\n",nt,N,t1-t0,(t1-t0)*1000.0/N);
    }
  }
  // threaded kernel-like: per-row omp for doing a real add on aa[0] row only? no, full.
  // Actually time the real kernel with per-row omp for (AVX not needed, scalar) to compare
  int best_nt=-1; double best=1e30;
  for(int k=0;k<4;k++){
    int nt=nts[k];
    double tb=1e30;
    for(int rep=0;rep<3;rep++){
      double t0=now_ms();
      #pragma omp parallel num_threads(nt)
      {
        for(int64_t i=1;i<LEN_2D;i++){
          #pragma omp for schedule(static)
          for(int64_t j=1;j<LEN_2D;j++){
            aa[i*LEN_2D+j]=aa[(i-1)*LEN_2D+(j-1)]+bb[i*LEN_2D+j];
          }
        }
      }
      double t1=now_ms(); if(t1-t0<tb)tb=t1-t0;
    }
    printf("threaded-kernel nt=%2d: %.1fms\n",nt,tb);
  }
  fflush(stdout);
}
