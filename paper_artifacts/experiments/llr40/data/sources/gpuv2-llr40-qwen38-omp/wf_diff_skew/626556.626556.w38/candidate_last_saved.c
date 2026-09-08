#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void v1(double *restrict a, int64_t N) {
  #pragma omp parallel
  {
    for (int64_t i = 1; i < N; ++i) {
      #pragma omp for schedule(static)
      for (int64_t j = 0; j < N - 1; ++j)
        a[i*N+j] = a[i*N+j] + a[(i-1)*N+j] + a[(i-1)*N+j+1];
    }
  }
}
static void v1p(double *restrict a, int64_t N) {
  #pragma omp parallel
  {
    for (int64_t i = 1; i < N; ++i) {
      #pragma omp for schedule(static)
      for (int64_t j = 0; j < N - 1; j += 1024)
        __builtin_prefetch(&a[i*N + j + 1024], 0, 1);
      #pragma omp for schedule(static)
      for (int64_t j = 0; j < N - 1; ++j)
        a[i*N+j] = a[i*N+j] + a[(i-1)*N+j] + a[(i-1)*N+j+1];
    }
  }
}
static void vb(double *restrict a, int64_t N) {  /* per-thread prev-row buffer */
  #pragma omp parallel
  {
    const int nth = omp_get_num_threads(), tid = omp_get_thread_num();
    const int64_t C = N - 1;
    const int64_t base = C / nth, rem = C % nth;
    const int64_t c0 = (int64_t)tid * base + (tid < rem ? tid : rem);
    const int64_t W = base + (tid < rem ? 1 : 0);
    if (W > 0) {
      double *pb = malloc((W + 1) * 8), *nb = malloc((W + 1) * 8);
      for (int64_t j = 0; j <= W; ++j) pb[j] = a[c0 + j];
      for (int64_t i = 1; i < N; ++i) {
        double *cr = a + i * N + c0;
        for (int64_t j = 0; j <= W; ++j) {
          double v = cr[j] + pb[j] + pb[j + 1];
          cr[j] = v; nb[j] = v;
        }
        double *t = pb; pb = nb; nb = t;
      }
      free(pb); free(nb);
    }
  }
}
void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N < 2) return;
  int chk = 0;
  #pragma omp target map(to: a[1]) map(tofrom: chk)
  { chk = 1 + (a[0] == 0.0); }
  double t0 = omp_get_wtime(); v1(a, N); double t1 = omp_get_wtime();
  printf("v1  24thr     : %8.3f ms\n", (t1-t0)*1e3);
  t0 = omp_get_wtime(); v1p(a, N); t1 = omp_get_wtime();
  printf("v1p +prefetch : %8.3f ms\n", (t1-t0)*1e3);
  t0 = omp_get_wtime(); vb(a, N); t1 = omp_get_wtime();
  printf("vb  prevbuf   : %8.3f ms\n", (t1-t0)*1e3);
  fflush(stdout);
}
