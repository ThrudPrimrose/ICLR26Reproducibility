#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  const int64_t N = LEN_2D * LEN_2D;
  const int do_b = (K > 0);
  const int nt = omp_get_max_threads();

  FILE *f = fopen("/proc/cpuinfo", "r");
  char line[512];
  int shown = 0, cores = 0;
  if (f) {
    while (fgets(line, sizeof line, f)) {
      if (strncmp(line, "model name", 10) == 0 && shown < 1) { printf("%s\n", line); shown = 1; }
      if (strncmp(line, "cpu cores", 9) == 0 && shown) { cores = atoi(line + 10); break; }
    }
    fclose(f);
  }
  printf("nt=%d cpu_cores=%d\n", nt, cores);

  /* scratch: 32 MB per thread, for write-BW probes */
  const int64_t per = 32 << 20; /* doubles */
  double *scratch = (double *)malloc((size_t)nt * per * sizeof(double));
  if (!scratch) { printf("malloc failed\n"); }

  /* Exp A: full b = src+1 pass, threaded */
  double tA0 = omp_get_wtime();
  if (do_b) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
      const int64_t base = i * LEN_2D;
#pragma omp simd
      for (int64_t j = 0; j < LEN_2D; ++j) b[base + j] = src[base + j] + 1.0;
    }
  }
  double tA1 = omp_get_wtime();

  /* Exp B: plain write to scratch, threaded */
  double tB0 = omp_get_wtime();
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    __m512d v = _mm512_set1_pd(1.0);
    for (int64_t k = 0; k < per; k += 8) ((__m512d *)(scratch + (int64_t)tid * per))[k / 8] = v;
  }
  double tB1 = omp_get_wtime();

  /* Exp C: NT write to scratch, threaded */
  double tC0 = omp_get_wtime();
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    __m512d v = _mm512_set1_pd(1.0);
    for (int64_t k = 0; k < per; k += 8) _mm512_stream_pd(scratch + (int64_t)tid * per + k, v);
  }
  _mm_sfence();
  double tC1 = omp_get_wtime();

  /* Exp D: single-thread plain read+write on the real arrays */
  double tD0 = omp_get_wtime();
  if (do_b) {
    for (int64_t k = 0; k < N; k += 8)
      _mm512_store_pd(b + k, _mm512_add_pd(_mm512_load_pd(src + k), _mm512_set1_pd(1.0)));
  }
  double tD1 = omp_get_wtime();

  /* Exp E: single-thread NT read+write on the real arrays */
  double tE0 = omp_get_wtime();
  if (do_b) {
    for (int64_t k = 0; k < N; k += 8)
      _mm512_stream_pd(b + k, _mm512_add_pd(_mm512_load_pd(src + k), _mm512_set1_pd(1.0)));
  }
  _mm_sfence();
  double tE1 = omp_get_wtime();

  /* correct a pass: cond rows only, threaded */
  int npos = 0;
  for (int64_t i = 0; i < LEN_2D; ++i) npos += (cond[i] > 0.0);
  double tF0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < LEN_2D; ++i) {
    const int64_t base = i * LEN_2D;
    if (cond[i] > 0.0) {
#pragma omp simd
      for (int64_t j = 0; j < LEN_2D; ++j) a[base + j] = src[base + j] * 2.0;
    }
  }
  double tF1 = omp_get_wtime();

  double gbA = 2.0 * (double)N * 8.0 / 1e9;
  double gbB = (double)nt * per * 8.0 / 1e9;
  printf("N=%ld npos=%d\n", (long)N, npos);
  printf("A multithread rw full:   %8.3f ms  %8.1f GB/s\n", (tA1 - tA0) * 1e3, do_b ? gbA / (tA1 - tA0) : 0);
  printf("B plain write scratch:   %8.3f ms  %8.1f GB/s\n", (tB1 - tB0) * 1e3, gbB / (tB1 - tB0));
  printf("C NT write scratch:      %8.3f ms  %8.1f GB/s\n", (tC1 - tC0) * 1e3, gbB / (tC1 - tC0));
  printf("D 1-thread rw full:      %8.3f ms  %8.1f GB/s\n", (tD1 - tD0) * 1e3, do_b ? gbA / (tD1 - tD0) : 0);
  printf("E 1-thread NT rw full:   %8.3f ms  %8.1f GB/s\n", (tE1 - tE0) * 1e3, do_b ? gbA / (tE1 - tE0) : 0);
  printf("F a-pass cond rows:      %8.3f ms  %8.1f GB/s\n", (tF1 - tF0) * 1e3, 16.0 * (double)npos * LEN_2D / 1e9 / (tF1 - tF0));
  free(scratch);
  fflush(NULL);
}
