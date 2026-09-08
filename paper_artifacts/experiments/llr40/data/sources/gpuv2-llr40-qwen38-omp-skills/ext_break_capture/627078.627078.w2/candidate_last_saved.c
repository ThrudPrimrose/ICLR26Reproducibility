/* probe: judge node characteristics */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <time.h>

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
  const double k = 1;
  out_index[0] = -1;
  out_value[0] = -1.0;
  for (int64_t i = 0; i < LEN_1D; ++i) {
    if (a[i] > k) { out_index[0] = i; out_value[0] = a[i]; break; }
  }

  int64_t n = LEN_1D;
  long nproc = sysconf(_SC_NPROCESSORS_ONLN);
  int nt = omp_get_max_threads();
  const char *env = getenv("OMP_NUM_THREADS");
  printf("n=%lld (=%.1f MB) nproc=%ld omp_max_threads=%d OMP_NUM_THREADS=%s\n",
         (long long)n, n*8.0/1e6, nproc, nt, env ? env : "(unset)");
  fflush(stdout);
  int ndev = omp_get_num_devices();
  int ondev = 0;
  #pragma omp target map(from: ondev)
  ondev = !omp_is_initial_device();
  printf("omp_get_num_devices=%d on_device=%d\n", ndev, ondev);
  fflush(stdout);

  if (n > 1000000) {
    int64_t m = n;
    if (m > 64ll*1024*1024) m = 64ll*1024*1024; /* 512 MB read window */
    double t0 = now_ms(), sum = 0.0;
    for (int64_t i = 0; i < m; i += 8) sum += a[i] + a[i+4];
    double t1 = now_ms();
    printf("cpu read 1 thread: %.1f GB in %.2f ms = %.1f GB/s (sum=%a)\n", m*8/1e9, t1-t0, m*8/((t1-t0)*1e6), sum);
    fflush(stdout);
    t0 = now_ms();
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (int64_t i = 0; i < m; i += 8) sum += a[i] + a[i+4];
    t1 = now_ms();
    printf("cpu read all threads: %.1f GB in %.2f ms = %.1f GB/s\n", m*8/1e9, t1-t0, m*8/((t1-t0)*1e6));
    fflush(stdout);

    t0 = now_ms();
    #pragma omp target data map(to: a[0:n]) { }
    t1 = now_ms();
    printf("gpu copy full %.1f GB (map(to:)): %.2f ms = %.1f GB/s\n", n*8/1e9, t1-t0, n*8/((t1-t0)*1e6));
    fflush(stdout);
    t0 = now_ms();
    #pragma omp target data map(to: a[0:n]) { }
    t1 = now_ms();
    printf("gpu copy full again: %.2f ms = %.1f GB/s\n", t1-t0, n*8/((t1-t0)*1e6));
    fflush(stdout);

    t0 = now_ms();
    int64_t pos = -2;
    #pragma omp target data map(to: a[0:n]) {
      #pragma omp target teams distribute parallel for reduction(min: pos)
      for (int64_t i = 0; i < n; ++i) if (a[i] > 1.0) pos = i;
    }
    t1 = now_ms();
    printf("gpu copy+scan whole (reduction min): pos=%lld total=%.2f ms\n", (long long)pos, t1-t0);
    fflush(stdout);
    (void)pos;
  }
}
