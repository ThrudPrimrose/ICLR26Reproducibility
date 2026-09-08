#include <stdint.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static double bench_store(double *p, size_t n, int nt_store, int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    double t0 = omp_get_wtime();
    if (nt_store) {
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < (int64_t)n; i += 4)
        _mm256_stream_pd(p + i, _mm256_set1_pd(1.0));
    } else {
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < (int64_t)n; ++i)
        p[i] = 1.0;
    }
    double t1 = omp_get_wtime();
    if (t1 - t0 < best) best = t1 - t0;
  }
  return best;
}

static double bench_fillread(double *w, size_t n) {
  double best = 1e30;
  for (int r = 0; r < 3; ++r) {
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)n; ++i)
      w[i] = w[i] + 1.0;
    double t1 = omp_get_wtime();
    if (t1 - t0 < best) best = t1 - t0;
  }
  return best;
}

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 0) return;

#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < N; ++i)
    a[i] += b[i] * c[i];

  const int64_t nt = omp_get_max_threads();
  int64_t W = (N + (nt < 1 ? 1 : nt) - 1) / (nt < 1 ? 1 : nt);
  W = (W + 7) & ~(int64_t)7;
  if (W < 64) W = 64;
  if (W > N) W = N;

#pragma omp parallel for schedule(static)
  for (int64_t i0 = 0; i0 < N; i0 += W) {
    int64_t i1 = i0 + W;
    if (i1 > N) i1 = N;
    for (int64_t j = 1; j < N; ++j) {
      double *restrict ra = aa + j * N;
      const double *restrict pa = aa + (j - 1) * N;
      const double *restrict rb = bb + j * N;
      for (int64_t i = i0; i < i1; ++i)
        ra[i] = pa[i] + rb[i] * a[i];
    }
  }

  /* ---- probe: bandwidth microbenchmarks, once, on first call ---- */
  static int done = 0;
  if (done) return;
  done = 1;
  size_t n = (size_t)N * N;
  double *w = (double *)aligned_alloc(256, 256 * (n / 32 + 1));
  fprintf(stdout, "PROBE2 N=%ld nt=%d procs=%d affinity=%d\n", (long)N, omp_get_max_threads(),
          omp_get_num_procs(), -1);
  double t_store = bench_store(w, n, 0, 2);
  double t_nt    = bench_store(w, n, 1, 2);
  double t_fillr = bench_fillread(w, n);
  fprintf(stdout, "PROBE2 store=%.4f ms  nt_store=%.4f ms  fillread=%.4f ms\n",
          t_store * 1e3, t_nt * 1e3, t_fillr * 1e3);
  fprintf(stdout, "PROBE2 GB/s store=%.1f nt=%.1f fillread=%.1f\n",
          8.0 * n / t_store / 1e9, 8.0 * n / t_nt / 1e9, 16.0 * n / t_fillr / 1e9);
  fflush(stdout);
  free(w);
}
