#include <stdint.h>
#include <stdio.h>
#include <omp.h>

#define OMP_DEV 1

#pragma omp declare target
static double *da = 0;
static double *db = 0;
static int32_t *dip = 0;
#pragma omp end declare target

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b,
                       const int32_t *restrict ip, const int64_t LEN_1D) {
  int64_t N = LEN_1D;
  if (N <= 0) return;
  if (da == 0) {
    da = (double *)omp_target_alloc(sizeof(double) * N, 64);
    db = (double *)omp_target_alloc(sizeof(double) * N, 64);
    dip = (int32_t *)omp_target_alloc(sizeof(int32_t) * N, 64);
    omp_target_memcpy(da, a, N * 8, 0, 0, OMP_DEV, 0);
    omp_target_memcpy(db, b, N * 8, 0, 0, OMP_DEV, 0);
    omp_target_memcpy(dip, ip, N * 4, 0, 0, OMP_DEV, 0);
  }
  double t0 = omp_get_wtime();
  omp_target_memcpy(da, a, N * 8, 0, 0, OMP_DEV, 0);
  double t1 = omp_get_wtime();
  for (int r = 0; r < 5; ++r) {
    #pragma omp target map(to: da, db, dip, N)
    {
      #pragma omp teams distribute parallel for
      for (int64_t i = 0; i < N; ++i) { da[i] += db[dip[i]] * 2.0; }
    }
  }
  double t2 = omp_get_wtime();
  omp_target_memcpy(a, da, N * 8, 0, 0, 0, OMP_DEV);
  double t3 = omp_get_wtime();
  printf("N=%lld copyin=%.3f compute=%.3f copyout=%.3f total=%.3f ms\n",
         (long long)N, (t1-t0)*1e3, (t2-t1)*2e2, (t3-t2)*1e3, (t3-t0)*1e3);
  fflush(stdout);
}
