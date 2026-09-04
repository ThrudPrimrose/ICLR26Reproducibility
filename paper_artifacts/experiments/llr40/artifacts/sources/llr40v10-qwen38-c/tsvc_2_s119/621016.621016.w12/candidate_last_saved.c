#include <stdint.h>
#include <omp.h>

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  if (N < 2) return;
  #pragma omp parallel
  {
    const int nt = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int64_t C = N - 1; /* j = 1..N-1  ->  k = j-1 = 0..N-2 */
    const int64_t c0 = (C * tid) / nt;
    const int64_t c1 = (C * (tid + 1)) / nt;
    for (int64_t i = 1; i < N; ++i) {
      double *const r = aa + i * N;
      const double *const rm1 = aa + (i - 1) * N;
      const double *const b = bb + i * N;
      for (int64_t k = c0; k < c1; ++k) {
        r[k + 1] = rm1[k] + b[k + 1];
      }
    }
  }
}
