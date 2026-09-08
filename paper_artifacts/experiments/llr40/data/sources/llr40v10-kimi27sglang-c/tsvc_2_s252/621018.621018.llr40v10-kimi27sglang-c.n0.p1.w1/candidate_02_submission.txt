#include <stdint.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;

  const int64_t BS = 4096;  // blocks fit in L1 cache
  const int64_t nblocks = (LEN_1D + BS - 1) / BS;

  #pragma omp parallel for schedule(static)
  for (int64_t blk = 0; blk < nblocks; ++blk) {
    const int64_t lo = blk * BS;
    const int64_t hi = (lo + BS < LEN_1D) ? lo + BS : LEN_1D;

    double buf[4096];

    // compute products into buffer
    for (int64_t i = lo; i < hi; ++i) {
      buf[i - lo] = b[i] * c[i];
    }

    // write first element with boundary value
    if (lo == 0) {
      a[0] = buf[0];
    } else {
      a[lo] = buf[0] + b[lo - 1] * c[lo - 1];
    }

    // shifted sums within block
    for (int64_t i = lo + 1; i < hi; ++i) {
      a[i] = buf[i - lo] + buf[i - lo - 1];
    }
  }
}
