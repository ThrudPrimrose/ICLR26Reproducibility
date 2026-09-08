#include <stdint.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t N) {
  if (N <= 1) return;
  #pragma omp parallel
  {
    int64_t tid = omp_get_thread_num();
    int64_t nt = omp_get_num_threads();
    int64_t i0 = (N * tid) / nt;
    int64_t i1 = (N * (tid + 1)) / nt;
    const double *M = aa;
    for (int64_t j = 1; j < N; j++) {
      const double *A = aa + (j - 1) * N;
      double *Aw = aa + j * N;
      const double *B = bb + j * N;
      const double *C = cc + j * N;
      for (int64_t i = i0; i < i1; i++)
        if (M[i] > 0.0)
          Aw[i] = A[i] + B[i] * C[i];
    }
  }
}
