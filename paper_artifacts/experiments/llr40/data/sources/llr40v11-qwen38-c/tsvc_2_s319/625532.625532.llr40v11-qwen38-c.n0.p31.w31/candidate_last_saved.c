#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

static int printed = 0;

static size_t dumpfile(const char *path, size_t maxlines) {
  FILE *f = fopen(path, "r");
  if (!f) { printf("%s:open_fail ", path); return 0; }
  char line[4096];
  size_t n = 0;
  while (fgets(line, sizeof line, f) && n < maxlines) {
    size_t r = strlen(line);
    if (r == 0) break;
    while (r > 0 && (line[r-1] == '\n' || line[r-1] == '\r')) line[--r] = 0;
    fwrite(line, 1, r, stdout);
    putchar(' ');
    n++;
  }
  fclose(f);
  return n;
}

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
  (void)a; (void)b; (void)d; (void)e;
  if (!printed) {
    printed = 1;
    printf("c_ptr=%p\n", (void *)c);
    printf("STATUS:\n");
    dumpfile("/proc/self/status", 300);
    printf("\nNUMA:");
    dumpfile("/proc/self/numa_maps", 200);
    printf("\n");
    fflush(stdout);
  }
  const int64_t N = LEN_1D;
  double total = 0.0;
  #pragma omp parallel reduction(+:total)
  {
    double s0=0,s1=0,t0=0,t1=0;
    #pragma omp for schedule(static) nowait
    for (int64_t i = 0; i < N; ++i) {
      double v = c[i] + d[i];
      double u = c[i] + e[i];
      a[i] = v; b[i] = u;
      if ((i & 1) == 0) { s0 += v; t0 += u; } else { s1 += v; t1 += u; }
    }
    total += (s0 + s1) + (t0 + t1);
  }
  b[0] = total;
}
