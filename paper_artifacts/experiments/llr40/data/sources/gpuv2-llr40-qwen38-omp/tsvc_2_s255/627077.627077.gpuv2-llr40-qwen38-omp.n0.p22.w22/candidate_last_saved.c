#include <stdint.h>

/* s255: a[i] = (b[i] + x + y) * 0.333, where x,y are the two previous
 * elements of b, wrapping at the end:
 *   x = b[(i-1) mod n], y = b[(i-2) mod n]
 * The sequential dependency chain in the reference is therefore just an
 * elementwise gather with two wrap-around entries, which parallelizes.
 *
 * Small n: run on the host (the offload round trip would not amortize).
 * Large n: elementwise on the device with the two wrap entries handled on
 * the host side afterwards.  Only a[2:n] crosses the wire: the device image
 * of a[0:2] is never written and would overwrite the host values if mapped. */

static void host_run(double *restrict a, const double *restrict b,
                     const int64_t n) {
  a[0] = (b[0] + b[n-1] + b[n-2]) * 0.333;
  a[1] = (b[1] + b[0] + b[n-1]) * 0.333;
  if (n <= (1 << 15)) {
    for (int64_t i = 2; i < n; i++)
      a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333;
  } else {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 2; i < n; i++)
      a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333;
  }
}

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b,
                      const int64_t n) {
  if (n < 2) return;
  if (n < (1 << 20)) {
    host_run(a, b, n);
    return;
  }
  #pragma omp target teams distribute parallel for map(to: b[0:n]) map(from: a[2:n-2])
  for (int64_t i = 2; i < n; i++)
    a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333;
  a[0] = (b[0] + b[n-1] + b[n-2]) * 0.333;
  a[1] = (b[1] + b[0] + b[n-1]) * 0.333;
}
