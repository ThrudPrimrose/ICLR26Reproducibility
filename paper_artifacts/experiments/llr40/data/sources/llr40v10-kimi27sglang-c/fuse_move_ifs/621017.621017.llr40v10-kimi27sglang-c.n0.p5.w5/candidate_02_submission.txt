#include <stdint.h>
#include <omp.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  #pragma omp parallel
  {
    if (K > 0) {
      #pragma omp for nowait
      for (int64_t i = 0; i < LEN_2D; ++i) {
        const int64_t base = i * LEN_2D;
        if (cond[i] > 0.0) {
          for (int64_t j = 0; j < LEN_2D; ++j) {
            const double s = src[base + j];
            b[base + j] = s + 1.0;
            a[base + j] = s * 2.0;
          }
        } else {
          for (int64_t j = 0; j < LEN_2D; ++j) {
            b[base + j] = src[base + j] + 1.0;
          }
        }
      }
    } else {
      #pragma omp for nowait
      for (int64_t i = 0; i < LEN_2D; ++i) {
        if (cond[i] > 0.0) {
          const int64_t base = i * LEN_2D;
          for (int64_t j = 0; j < LEN_2D; ++j) {
            a[base + j] = src[base + j] * 2.0;
          }
        }
      }
    }
  }
}
