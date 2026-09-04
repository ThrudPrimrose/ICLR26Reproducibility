/* TSVC tsvc_2 s255: the "serial" recurrence is a trap.
 * x tracks b[i] and y tracks b[i-1]; a[i] never feeds back.
 * So: a[0] = (b[0]+b[N-1]+b[N-2])*0.333
 *     a[1] = (b[1]+b[0]+b[N-1])*0.333
 *     a[i] = (b[i]+b[i-1]+b[i-2])*0.333  (i>=2)
 * Bit-identical to the reference (same fp op order per element). */
#include <stdint.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n <= 1) {
    if (n == 1) a[0] = (b[0] + b[0] + b[-1]) * 0.333;
    return;
  }
  a[0] = (b[0] + b[n-1] + b[n-2]) * 0.333;
  a[1] = (b[1] + b[0] + b[n-1]) * 0.333;
  for (int64_t i = 2; i < n; i++)
    a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333;
}
