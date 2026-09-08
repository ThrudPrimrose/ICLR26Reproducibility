#include <stdint.h>
#include <stdio.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
  int64_t mn = ip[0], mx = ip[0];
  int64_t inrange = 0;
  int64_t sorted = 0;
  for (int64_t i = 0; i < LEN_1D; ++i) {
    if (ip[i] < mn) mn = ip[i];
    if (ip[i] > mx) mx = ip[i];
    if (ip[i] >= 0 && ip[i] < LEN_1D) inrange++;
    if (i > 0 && ip[i] >= ip[i-1]) sorted++;
  }
  for (int64_t i = 0; i < LEN_1D; ++i) a[i] = b[ip[i]];
  printf("LEN_1D=%ld mn=%ld mx=%ld inrange=%ld sorted_frac=%.3f b0=%g\n",
         (long)LEN_1D, (long)mn, (long)mx, (long)inrange, (double)sorted/(LEN_1D-1), b[0]);
  fflush(NULL);
}
