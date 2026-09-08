#include <stdint.h>
#include <stdio.h>
#include <omp.h>
void versioned_distance_update_fp64(double *restrict a, double *restrict b, double *restrict c,
                                    const int64_t LEN_1D, const int64_t K,
                                    uint8_t *workspace, const int64_t workspace_size){
  (void)b;(void)c;(void)workspace;(void)workspace_size;
  printf("PROBE LEN_1D=%lld K=%lld omp_max_threads=%d\n",
         (long long)LEN_1D,(long long)K, omp_get_max_threads());
  fflush(stdout);
  for (int64_t i = K; i < LEN_1D; ++i) a[i] = 0.75 * a[i-K] + b[i]*c[i];
}
