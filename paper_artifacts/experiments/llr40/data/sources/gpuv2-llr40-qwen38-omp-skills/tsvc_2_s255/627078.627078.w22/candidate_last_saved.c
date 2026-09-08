/* TSVC s255: a[i] = (b[i] + b[i-1] + b[i-2]) * 0.333, indices wrap (circular).
 * The reference's x/y carry is a false dependence (x==b[i-1], y==b[i-2]), so
 * after peeling the two wraparound elements the loop is a plain parallel
 * stencil. The peel runs on the host; one combined target region moves b in
 * and a out and does the stencil on the device. */

#include <stdint.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D)
{
  if (LEN_1D < 2)
    return; /* reference reads b[-1], b[-2] for LEN<2: UB there, nothing to do */

  /* the two wraparound elements, computed where the data already is */
  a[0] = (b[0] + b[LEN_1D - 1] + b[LEN_1D - 2]) * 0.333;
  a[1] = (b[1] + b[0] + b[LEN_1D - 1]) * 0.333;

  #pragma omp target teams distribute parallel for map(to: b[0:LEN_1D]) map(tofrom: a[2:LEN_1D - 2])
  for (int64_t i = 2; i < LEN_1D; i++)
    a[i] = (b[i] + b[i - 1] + b[i - 2]) * 0.333;
}
