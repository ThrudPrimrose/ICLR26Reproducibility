#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

static void probe_env(void) {
  FILE *f = fopen("/proc/self/status", "r");
  char line[512];
  if (f) {
    while (fgets(line, sizeof line, f)) {
      if (line[0] == 'C' && (line[1]=='p' || line[1]=='M')) { printf("%s", line); }
    }
    fclose(f);
  }
  printf("PROBE2 OMP max_threads=%d\n", omp_get_max_threads());
  fflush(stdout);
}
static __thread int g_done = 0;

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
  if (!g_done) { g_done = 1; probe_env(); }
  printf("PROBE2 LEN_2D=%lld a[0..3]= %.6g %.6g %.6g %.6g\n",
         (long long)LEN_2D, a[0], a[1], a[2], a[3]);
  double mx = 0.0;
  for (int64_t k = 0; k < LEN_2D * LEN_2D; ++k) {
    double v = a[k] < 0 ? -a[k] : a[k];
    if (v > mx) mx = v;
  }
  printf("PROBE2 input maxabs=%.9g\n", mx);
  fflush(stdout);
  for (int64_t i = 1; i < LEN_2D; ++i)
    for (int64_t j = i; j < LEN_2D; ++j)
      a[i * LEN_2D + j] = a[i * LEN_2D + j] + a[(i - 1) * LEN_2D + j] + a[i * LEN_2D + (j - 1)];
}
