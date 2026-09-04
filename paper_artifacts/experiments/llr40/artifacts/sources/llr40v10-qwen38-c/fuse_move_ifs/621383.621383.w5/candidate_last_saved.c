#include <stdint.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b, const double *restrict cond, const double *restrict src,
                        const int64_t K, const int64_t LEN_2D) {
  if (K > 0) {
    for (int64_t i = 0; i < LEN_2D; ++i) {
      const double *restrict s = src + i * LEN_2D;
      double *restrict ap = a + i * LEN_2D;
      double *restrict bp = b + i * LEN_2D;
      if (cond[i] > 0.0) {
        for (int64_t j = 0; j < LEN_2D; ++j) {
          ap[j] = s[j] * 2.0;
          bp[j] = s[j] + 1.0;
        }
      } else {
        for (int64_t j = 0; j < LEN_2D; ++j) bp[j] = s[j] + 1.0;
      }
    }
  } else {
    for (int64_t i = 0; i < LEN_2D; ++i) {
      const double *restrict s = src + i * LEN_2D;
      double *restrict ap = a + i * LEN_2D;
      if (cond[i] > 0.0) {
        for (int64_t j = 0; j < LEN_2D; ++j) ap[j] = s[j] * 2.0;
      }
    }
  }
}
