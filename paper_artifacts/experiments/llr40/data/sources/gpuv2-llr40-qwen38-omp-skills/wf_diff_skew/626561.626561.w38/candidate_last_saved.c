#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <omp.h>

static void serial_v(double *a, int64_t N) {
  for (int64_t i = 1; i < N; i++) {
    double *row = a + i * N, *prev = a + (i - 1) * N;
    for (int64_t j = 0; j < N - 1; j++)
      row[j] = row[j] + prev[j] + prev[j + 1];
  }
}

static void flagpipe(double *a, int64_t N, int64_t T) {
  int64_t JMAX = N - 1;
  int64_t *J = malloc((T + 1) * sizeof(int64_t));
  for (int64_t t = 0; t <= T; t++) J[t] = t * JMAX / T;
  static _Atomic long done[64];
  for (long t = 0; t < T; t++) atomic_store_explicit(&done[t], 0, memory_order_relaxed);
  #pragma omp parallel num_threads(T)
  {
    long t = omp_get_thread_num();
    int64_t j0 = J[t], j1 = J[t + 1];
    long nb = (t + 1 < T) ? (t + 1) : -1;
    for (int64_t i = 1; i < N; i++) {
      if (nb >= 0) {
        long v;
        while ((v = atomic_load_explicit(&done[nb], memory_order_acquire)) < i - 1) { }
      }
      double *row = a + i * N, *prev = a + (i - 1) * N;
      for (int64_t j = j0; j < j1; j++)
        row[j] = row[j] + prev[j] + prev[j + 1];
      atomic_store_explicit(&done[t], (long)i, memory_order_release);
    }
  }
  free(J);
}

static void rowbar(double *a, int64_t N) {
  #pragma omp parallel num_threads(24)
  for (int64_t i = 1; i < N; i++) {
    #pragma omp for
    for (int64_t j = 0; j < N - 1; j++)
      a[i * N + j] = a[i * N + j] + a[(i - 1) * N + j] + a[(i - 1) * N + j + 1];
  }
}

static void devtouch(double *a) {
  #pragma omp target map(tofrom: a[0:8])
    {
      #pragma omp teams distribute parallel for simd
      for (int64_t j = 0; j < 8; j++) a[j] = a[j] * 1.0;
    }
}

static const char *names[6] = { "serial", "flag24", "flag12", "flag8", "flag16", "rowbar24" };

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N < 2) return;
  double *base = malloc((size_t)N * N * sizeof(double));
  memcpy(base, a, (size_t)N * N * sizeof(double));

  printf("BAT N=%lld nt=%d ndev=%d\n", (long long)N, omp_get_max_threads(), omp_get_num_devices()); fflush(stdout);
  double t0, t1;
  for (int k = 0; k < 6; k++) {
    memcpy(a, base, (size_t)N * N * sizeof(double));
    devtouch(a);
    t0 = omp_get_wtime();
    if (k == 0) serial_v(a, N);
    else if (k == 1) flagpipe(a, N, 24);
    else if (k == 2) flagpipe(a, N, 12);
    else if (k == 3) flagpipe(a, N, 8);
    else if (k == 4) flagpipe(a, N, 16);
    else rowbar(a, N);
    t1 = omp_get_wtime();
    printf("BAT %s: %.1f ms\n", names[k], (t1 - t0) * 1e3);
  }
  fflush(stdout);
  free(base);
}
