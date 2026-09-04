#include <stdint.h>
#include <omp.h>

/* fuse_diamond: out[i] = (a[i]*a[i] + 1.0) * (a[i]*a[i] - 1.0)
 *
 * Bit-exact with the numpy oracle: each rounding step (mul, +1, -1, mul) is a
 * separate IEEE operation. fp-contract=off forbids FMA contraction, which would
 * otherwise merge the square with the +/-1 and shift results by up to 1 ulp.
 * The loop auto-vectorizes (ZMM, unrolled) since it is a plain restrict loop. */
static __attribute__((optimize("fp-contract=off"), noinline)) void
fd_chunk(const double *restrict a, double *restrict out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    double t = a[i] * a[i];
    out[i] = (t + 1.0) * (t - 1.0);
  }
}

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  if (LEN_1D > (1 << 20)) {
    #pragma omp parallel
    {
      int nt = omp_get_num_threads();
      int64_t base = (nt > 0) ? LEN_1D / nt : LEN_1D;
      int64_t rem = (nt > 0) ? LEN_1D - base * nt : 0;
      int tid = omp_get_thread_num();
      int64_t beg = (int64_t)tid * base + (((int64_t)tid < rem) ? (int64_t)tid : rem);
      int64_t cnt = base + (((int64_t)tid < rem) ? 1 : 0);
      fd_chunk(a + beg, out + beg, cnt);
    }
  } else {
    fd_chunk(a, out, LEN_1D);
  }
}
