#include <stdint.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N <= 384) {
    for (int64_t j = 1; j < N; j++) {
      const double *const prow = aa + (j - 1) * N;
      const double *const brow = bb + j * N;
      const double *const crow = cc + j * N;
      double *const arow = aa + j * N;
      for (int64_t i = 0; i < N; i++) {
        if (aa[i] > 0.0) arow[i] = prow[i] + brow[i] * crow[i];
      }
    }
    return;
  }
#pragma omp parallel
  {
    const int64_t nt  = omp_get_num_threads();
    const int64_t tid = omp_get_thread_num();
    const int64_t i0 = (N * tid) / nt;
    const int64_t i1 = (N * (tid + 1)) / nt;
    const int64_t jend = N - 2;
    for (int64_t j = 1; j < jend; j++) {
      const double *const prow = aa + (j - 1) * N;
      const double *const brow = bb + j * N;
      const double *const crow = cc + j * N;
      double *const arow = aa + j * N;
      const double *const pbrow = bb + (j + 2) * N;
      const double *const pcrow = cc + (j + 2) * N;
      for (int64_t i = i0; i < i1; i++) {
        __builtin_prefetch(pbrow + i, 0, 2);
        __builtin_prefetch(pcrow + i, 0, 2);
        if (aa[i] > 0.0) arow[i] = prow[i] + brow[i] * crow[i];
      }
    }
    for (int64_t j = jend; j < N; j++) {
      const double *const prow = aa + (j - 1) * N;
      const double *const brow = bb + j * N;
      const double *const crow = cc + j * N;
      double *const arow = aa + j * N;
      for (int64_t i = i0; i < i1; i++) {
        if (aa[i] > 0.0) arow[i] = prow[i] + brow[i] * crow[i];
      }
    }
  }
}
