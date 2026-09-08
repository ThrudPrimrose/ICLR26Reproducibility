#include <stdint.h>
#include <omp.h>

/* TSVC tsvc_2_5 fuse_stencil_through_transient, fp64, single invocation.
 *
 * out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])
 * for i in [1, LEN_1D-3].  The loop is fully parallel.
 *
 * For large inputs the region is offloaded.  The map round trip is the cost,
 * so we move only what must move: `a` in, the written part of `out` back.
 * The three elements the stencil never writes (out[0], out[n-2], out[n-1])
 * are kept on the host: saved before, restored after, which also shields
 * them from any 16-byte-granule overrun of the misaligned `out[1:hi]` copy.
 */

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n <= 3) return;
  const int64_t hi = n - 2; /* i runs 1 .. n-3 */

  if (n >= (1 << 20)) {
    const double b0 = out[0], bn2 = out[n - 2], bn1 = out[n - 1];
#pragma omp target map(to: a[0:n]) map(from: out[1:hi])
#pragma omp teams distribute parallel for simd
    for (int64_t i = 1; i < hi; ++i) {
      out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
    out[0] = b0;
    out[n - 2] = bn2;
    out[n - 1] = bn1;
  } else {
    for (int64_t i = 1; i < hi; ++i) {
      out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
  }
}
