#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  if (LEN_2D <= 1) return;
  #pragma omp parallel
  {
    const int nt = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int64_t base = LEN_2D / nt, rem = LEN_2D % nt;
    const int64_t i0 = tid * base + (tid < rem ? tid : rem);
    const int64_t i1 = i0 + base + (tid < rem ? 1 : 0);
    for (int64_t j = 1; j < LEN_2D; ++j) {
      double *restrict p = aa + j * LEN_2D;
      const double *restrict q = aa + (j - 1) * LEN_2D;
      const double *restrict b = bb + j * LEN_2D;
      for (int64_t i = i0; i < i1; ++i) {
        p[i] = q[i] + b[i];
      }
    }
  }
}
