#include <stdint.h>
#include <omp.h>

void wf_triangular_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;

  if (N < 512) {
    for (int64_t i = 1; i < N; ++i) {
      for (int64_t j = i; j < N; ++j) {
        a[i * N + j] = a[i * N + j] + a[(i - 1) * N + j] + a[i * N + (j - 1)];
      }
    }
    return;
  }

  const int64_t B = 30;
  const int64_t NB = (N + B - 1) / B;

  #pragma omp parallel
  {
    for (int64_t db = 0; db <= 2 * NB - 2; ++db) {
      int64_t bi0 = db - (NB - 1);
      if (bi0 < 0) bi0 = 0;
      int64_t bi1 = db / 2;
      if (bi1 >= NB) bi1 = NB - 1;

      #pragma omp for schedule(static)
      for (int64_t bi = bi0; bi <= bi1; ++bi) {
        const int64_t bj = db - bi;
        const int64_t i0 = bi * B;
        const int64_t i1 = (bi + 1) * B - 1 < N ? (bi + 1) * B - 1 : N - 1;
        const int64_t j0 = bj * B;
        const int64_t j1 = (bj + 1) * B - 1 < N ? (bj + 1) * B - 1 : N - 1;
        const int64_t istart = i0 < 1 ? 1 : i0;

        if (bj == bi) {
          for (int64_t i = istart; i <= i1; ++i) {
            double *restrict row = a + i * N;
            const double *restrict nrow = a + (i - 1) * N;
            for (int64_t j = i; j <= j1; ++j) {
              row[j] = row[j] + nrow[j] + row[j - 1];
            }
          }
        } else {
          for (int64_t i = istart; i <= i1; ++i) {
            double *restrict row = a + i * N;
            const double *restrict nrow = a + (i - 1) * N;
            for (int64_t j = j0; j <= j1; ++j) {
              row[j] = row[j] + nrow[j] + row[j - 1];
            }
          }
        }
      }
    }
  }
}
