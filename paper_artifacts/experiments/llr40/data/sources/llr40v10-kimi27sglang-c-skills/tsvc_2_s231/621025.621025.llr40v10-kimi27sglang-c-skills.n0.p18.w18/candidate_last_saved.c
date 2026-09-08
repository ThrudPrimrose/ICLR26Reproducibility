#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
  #pragma omp parallel
  {
    const int64_t nthreads = omp_get_num_threads();
    const int64_t tid = omp_get_thread_num();
    const int64_t align = 8;
    const int64_t chunk = (((LEN_2D + nthreads - 1) / nthreads) + align - 1) & ~(align - 1);
    const int64_t i_start = tid * chunk;
    const int64_t i_end = (i_start + chunk < LEN_2D) ? (i_start + chunk) : LEN_2D;
    const int64_t len = i_end - i_start;

    for (int64_t j = 1; j < LEN_2D; ++j) {
      double *restrict a_prev = aa + (j - 1) * LEN_2D + i_start;
      double *restrict a_curr = aa + j * LEN_2D + i_start;
      const double *restrict b_curr = bb + j * LEN_2D + i_start;

      const uintptr_t off = (uintptr_t)a_curr & 63ULL;
      int64_t peel = (off == 0) ? 0 : (int64_t)((64ULL - off) >> 3);
      if (peel > len) peel = len;

      for (int64_t i = 0; i < peel; ++i) {
        a_curr[i] = a_prev[i] + b_curr[i];
      }

      a_prev += peel;
      a_curr += peel;
      b_curr += peel;
      const int64_t rem = len - peel;

      #pragma omp simd aligned(a_prev, a_curr, b_curr:64)
      for (int64_t i = 0; i < rem; ++i) {
        a_curr[i] = a_prev[i] + b_curr[i];
      }
    }
  }
}
