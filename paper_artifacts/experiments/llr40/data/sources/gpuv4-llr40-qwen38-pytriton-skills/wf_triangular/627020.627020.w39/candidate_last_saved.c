#include <stdint.h>
#include <omp.h>

/* 2D-blocked wavefront over the triangle j >= i.
 * Block (bi,bj) covers rows [bi*B,(bi+1)*B), cols [bj*B,(bj+1)*B).
 * Super-diagonal s = bi+bj: all blocks on it are independent (each depends
 * only on blocks with s-1). Within a block, rows are sequential scans. */

void wf_triangular_kernel_f64(double* __restrict a, int64_t N, int64_t B) {
  int64_t nb = (N + B - 1) / B;
  #pragma omp parallel
  {
    for (int64_t s = 0; s <= 2*nb - 2; ++s) {
      int64_t bi0 = s - (nb - 1); if (bi0 < 0) bi0 = 0;
      int64_t bi1 = s / 2;        if (bi1 > nb - 1) bi1 = nb - 1;
      #pragma omp for schedule(static)
      for (int64_t bi = bi0; bi <= bi1; ++bi) {
        int64_t bj = s - bi;
        int64_t i0 = bi * B;
        int64_t i1 = i0 + B; if (i1 > N) i1 = N;
        int64_t j0 = bj * B;
        int64_t j1 = j0 + B; if (j1 > N) j1 = N;
        for (int64_t i = i0; i < i1; ++i) {
          if (i == 0) continue;
          int64_t js = i > j0 ? i : j0;
          if (js >= j1) continue;
          double* row = a + i * N;
          double* up  = a + (i - 1) * N;
          double left = row[js - 1];
          for (int64_t j = js; j < j1; ++j) {
            double v = row[j] + up[j] + left;
            row[j] = v;
            left = v;
          }
        }
      }
    }
  }
}

void wf_triangular_kernel_f32(float* __restrict a, int64_t N, int64_t B) {
  int64_t nb = (N + B - 1) / B;
  #pragma omp parallel
  {
    for (int64_t s = 0; s <= 2*nb - 2; ++s) {
      int64_t bi0 = s - (nb - 1); if (bi0 < 0) bi0 = 0;
      int64_t bi1 = s / 2;        if (bi1 > nb - 1) bi1 = nb - 1;
      #pragma omp for schedule(static)
      for (int64_t bi = bi0; bi <= bi1; ++bi) {
        int64_t bj = s - bi;
        int64_t i0 = bi * B;
        int64_t i1 = i0 + B; if (i1 > N) i1 = N;
        int64_t j0 = bj * B;
        int64_t j1 = j0 + B; if (j1 > N) j1 = N;
        for (int64_t i = i0; i < i1; ++i) {
          if (i == 0) continue;
          int64_t js = i > j0 ? i : j0;
          if (js >= j1) continue;
          float* row = a + i * N;
          float* up  = a + (i - 1) * N;
          float left = row[js - 1];
          for (int64_t j = js; j < j1; ++j) {
            float v = row[j] + up[j] + left;
            row[j] = v;
            left = v;
          }
        }
      }
    }
  }
}
