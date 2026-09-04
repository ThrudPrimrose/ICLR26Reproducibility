#include <stdint.h>
#include <stdio.h>
void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  for (int64_t i = 0; i < LEN_1D - 1; ++i) {
    a[i] = a[i + 1] + b[i];
  }
  printf("PROBE_LEN_1D=%ld\n", (long)LEN_1D);
  fflush(stdout);
}
