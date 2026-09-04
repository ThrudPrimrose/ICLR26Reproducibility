#include <stdint.h>
#include <immintrin.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;

  const int64_t BS = 65536;
  const int64_t nblocks = (LEN_1D + BS - 1) / BS;

  #pragma omp parallel for schedule(static)
  for (int64_t blk = 0; blk < nblocks; ++blk) {
    const int64_t lo = blk * BS;
    const int64_t hi = (lo + BS < LEN_1D) ? lo + BS : LEN_1D;
    const int64_t n = hi - lo;

    // small per-thread buffer: for 64K elements = 512KB, fits L2
    double buf[65536];

    #pragma omp simd
    for (int64_t i = 0; i < n; ++i) {
      buf[i] = b[lo + i] * c[lo + i];
    }

    if (lo == 0) {
      a[lo] = buf[0];
    } else {
      a[lo] = buf[0] + b[lo - 1] * c[lo - 1];
    }

    #pragma omp simd
    for (int64_t i = 1; i < n; ++i) {
      a[lo + i] = buf[i] + buf[i - 1];
    }
  }
}
