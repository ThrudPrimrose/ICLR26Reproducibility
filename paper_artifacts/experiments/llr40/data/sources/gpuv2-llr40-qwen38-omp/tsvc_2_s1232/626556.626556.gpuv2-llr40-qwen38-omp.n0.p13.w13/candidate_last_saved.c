#include <stdint.h>
#include <omp.h>
#include <stdio.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D,
                       const int64_t VLEN) {
  const int64_t N = LEN_2D;
  int on_dev = -1;
  {
    #pragma omp target default(mappable) map(from: on_dev)
    {
      on_dev = !omp_is_initial_device();
      #pragma omp parallel for
      for (int64_t i = 0; i < N; ++i) {
        int64_t jmax = (VLEN > 0) ? (i / VLEN) : (N - 1);
        if (jmax >= N) jmax = N - 1;
        for (int64_t j = 0; j <= jmax; ++j) aa[i * N + j] = bb[i * N + j] + cc[i * N + j];
      }
    }
  }
  /* self-check 64 strided sample points */
  int bad = 0;
  for (int s = 0; s < 64; ++s) {
    int64_t i = (int64_t)(s + 1) * (N / 65);
    int64_t j = (i / 3) % N;
    if (j > i / VLEN && VLEN > 0) j = 0;
    if (aa[i * N + j] != bb[i * N + j] + cc[i * N + j]) bad++;
  }
  double t0 = omp_get_wtime();
  for (int k = 0; k < 12; ++k) {
    #pragma omp target default(mappable)
    {
      #pragma omp parallel for
      for (int64_t i = 0; i < N; ++i) {
        int64_t jmax = (VLEN > 0) ? (i / VLEN) : (N - 1);
        if (jmax >= N) jmax = N - 1;
        for (int64_t j = 0; j <= jmax; ++j) aa[i * N + j] = bb[i * N + j] + cc[i * N + j];
      }
    }
  }
  double t1 = omp_get_wtime();
  printf("MAPPABLE on_dev=%d bad=%d steady=%.4f ms/call\n", on_dev, bad, (t1 - t0) / 12 * 1000.0);
  fflush(stdout);
}
