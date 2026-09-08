#include <stdint.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  const int64_t N2 = N * N;

  if (K > 0) {
#pragma omp target data map(to: src[0:N2]) map(to: cond[0:N]) map(tofrom: a[0:N2]) map(tofrom: b[0:N2])
    {
#pragma omp target
#pragma omp teams distribute parallel for simd
      for (int64_t i = 0; i < N; ++i) {
        if (cond[i] > 0.0) {
          for (int64_t j = 0; j < N; ++j) {
            a[i * N + j] = src[i * N + j] * 2.0;
          }
        }
      }
#pragma omp target
#pragma omp teams distribute parallel for simd
      for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          b[i * N + j] = src[i * N + j] + 1.0;
        }
      }
    }
  } else {
#pragma omp target data map(to: src[0:N2]) map(to: cond[0:N]) map(tofrom: a[0:N2])
    {
#pragma omp target
#pragma omp teams distribute parallel for simd
      for (int64_t i = 0; i < N; ++i) {
        if (cond[i] > 0.0) {
          for (int64_t j = 0; j < N; ++j) {
            a[i * N + j] = src[i * N + j] * 2.0;
          }
        }
      }
    }
  }
}
