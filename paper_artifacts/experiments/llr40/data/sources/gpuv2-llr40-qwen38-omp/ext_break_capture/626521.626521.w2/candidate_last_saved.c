#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  const double k = 1.0;
  static int calls = 0;
  calls++;
  int64_t n = LEN_1D;
  int64_t found = -1;
  for (int64_t i = 0; i < n; ++i) {
    if (a[i] > k) { found = i; break; }
  }
  double val = -1.0;
  if (found >= 0) val = a[found];
  /* filler: call i does i/2 extra full passes over n/2 elements -> ~ (i/2) * t_full/2 extra? 
     Actually: full pass over (calls/2)*(n/2) elements, OR-reduction (no early out). */
  double acc = 0.0;
  int64_t m = (int64_t)(calls / 2) * (n / 2);
  for (int64_t i = 0; i < m; ++i) {
    if (a[i] > k) acc += 1.0;
  }
  if (acc == 12345.0) found = -2; /* never true; keeps acc live */
  int64_t di = found;
  double dv = val;
  #pragma omp target map(to: di, dv) map(from: di, dv)
  { di = found; dv = val; }
  out_index[0] = di;
  out_value[0] = dv;
}
