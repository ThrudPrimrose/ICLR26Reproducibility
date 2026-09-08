#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static double now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (double)ts.tv_sec*1e9+ts.tv_nsec; }

/* ---- row kernels (scalar; GCC vectorizes) ---- */
static void row_both(double *restrict arow, double *restrict brow, const double *restrict srow, int64_t N){
  for (int64_t j = 0; j < N; ++j) { const double s = srow[j]; arow[j] = s*2.0; brow[j] = s+1.0; }
}
static void row_b(double *restrict brow, const double *restrict srow, int64_t N){
  for (int64_t j = 0; j < N; ++j) { brow[j] = srow[j] + 1.0; }
}
/* 2 rows per thread for more MLP */
static void row_both2(double *restrict a0, double *restrict b0, const double *restrict s0,
                      double *restrict a1, double *restrict b1, const double *restrict s1, int64_t N){
  for (int64_t j = 0; j < N; ++j) {
    const double s0v = s0[j], s1v = s1[j];
    a0[j] = s0v*2.0; b0[j] = s0v+1.0;
    a1[j] = s1v*2.0; b1[j] = s1v+1.0;
  }
}
static void row_b2(double *restrict b0, const double *restrict s0,
                   double *restrict b1, const double *restrict s1, int64_t N){
  for (int64_t j = 0; j < N; ++j) { b0[j] = s0[j]+1.0; b1[j] = s1[j]+1.0; }
}

static void run_plain(double *a, double *b, const double *cond, const double *src, int64_t N, int nt){
  #pragma omp parallel num_threads(nt)
  {
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
      if (cond[i] > 0.0) row_both(a + i*N, b + i*N, src + i*N, N);
      else               row_b (b + i*N, src + i*N, N);
    }
  }
}
/* pair adjacent rows (i, i+1); last row alone */
static void run_pair(double *a, double *b, const double *cond, const double *src, int64_t N, int nt){
  #pragma omp parallel num_threads(nt)
  {
    #pragma omp for schedule(static)
    for (int64_t p = 0; p < (N+1)/2; ++p) {
      int64_t i0 = 2*p, i1 = 2*p+1;
      if (i1 >= N) { if (cond[i0] > 0.0) row_both(a+i0*N, b+i0*N, src+i0*N, N); else row_b(b+i0*N, src+i0*N, N); continue; }
      if (cond[i0] > 0.0 && cond[i1] > 0.0) row_both2(a+i0*N, b+i0*N, src+i0*N, a+i1*N, b+i1*N, src+i1*N, N);
      else if (cond[i0] > 0.0) { row_both(a+i0*N, b+i0*N, src+i0*N, N); row_b(b+i1*N, src+i1*N, N); }
      else if (cond[i1] > 0.0) { row_b(b+i0*N, src+i0*N, N); row_both(a+i1*N, b+i1*N, src+i1*N, N); }
      else row_b2(b+i0*N, src+i0*N, b+i1*N, src+i1*N, N);
    }
  }
}

static int selfcheck(double *a, double *b, const double *cond, const double *src, int64_t N){
  int bad = 0;
  const int64_t ii[6] = {0, 1, 37, N/3, N/2, N-1};
  const int64_t jj[6] = {0, 5, 12345, N/3, N-17, N-1};
  for (int x = 0; x < 6; ++x) for (int y = 0; y < 6; ++y) {
    int64_t i = ii[x], j = jj[y]; if (j >= N) j = N-1;
    double s = src[i*N+j];
    if (b[i*N+j] != s + 1.0) bad++;
    if (cond[i] > 0.0 && a[i*N+j] != s * 2.0) bad++;
  }
  return bad;
}

static int probed = 0;
void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (!probed) {
    probed = 1;
    char buf[4096]; int n = 0;
    n += snprintf(buf+n, sizeof(buf)-n, "PROBE maxthreads=%d K=%lld N=%lld\n", omp_get_max_threads(), (long long)K, (long long)N);
    struct V { const char *name; void (*fn)(double*,double*,const double*,const double*,int64_t,int); int nt; } V[] = {
      {"plain t24 ", run_plain, 24},
      {"plain t12 ", run_plain, 12},
      {"plain t32 ", run_plain, 32},
      {"plain t48 ", run_plain, 48},
      {"pair  t24 ", run_pair, 24},
      {"pair  t12 ", run_pair, 12},
      {"pair  t48 ", run_pair, 48},
    };
    for (unsigned v = 0; v < sizeof(V)/sizeof(V[0]); ++v) {
      double t0 = now_ns();
      V[v].fn(a,b,cond,src,N,V[v].nt);
      double t1 = now_ns();
      int bad = selfcheck(a,b,cond,src,N);
      n += snprintf(buf+n, sizeof(buf)-n, "PROBE %s %.2f ms %s\n", V[v].name, (t1-t0)/1e6, bad? "BAD!!":"ok");
    }
    fwrite(buf,1,n,stdout); fflush(stdout);
  }
  if (K > 0) {
    run_plain(a,b,cond,src,N,24);
  } else {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
      if (cond[i] > 0.0) {
        double *restrict arow = a + i * N;
        const double *restrict srow = src + i * N;
        for (int64_t j = 0; j < N; ++j) { arow[j] = srow[j] * 2.0; }
      }
    }
  }
}
