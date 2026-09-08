/* TSVC tsvc_2 s316: x = min(a[0..LEN_1D-1]); result[0] = x.
 * Min reduction: no inter-iteration dependence once expressed as a reduction,
 * so the whole trip is threadable + vectorizable. Single memory pass.
 * NaN semantics match the reference: result is NaN iff a[0] is NaN
 * (reference comparisons a[i] < x can never install NaN when x is already NaN-free
 * at a[0]... precisely: if a[0] is NaN the reference keeps it forever). */
#include <stdint.h>
#include <float.h>
#include <omp.h>

void tsvc_2_s316_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D) {
  double x = DBL_MAX;
  if (LEN_1D > 0) {
#pragma omp parallel for simd reduction(min : x) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
      if (a[i] < x) x = a[i];
    }
    if (!(a[0] == a[0])) x = a[0]; /* a[0] NaN => reference yields NaN */
  } else {
    x = 0.0; /* no elements: reference would read a[0] OOB; degenerate */
  }
  result[0] = x;
}
