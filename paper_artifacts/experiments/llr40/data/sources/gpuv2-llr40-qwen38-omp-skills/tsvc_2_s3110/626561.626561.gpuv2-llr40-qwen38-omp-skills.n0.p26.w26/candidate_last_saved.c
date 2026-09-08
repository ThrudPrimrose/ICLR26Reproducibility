#include <stdint.h>
#include <limits.h>
#include <omp.h>

typedef struct { double m; int64_t i; } mx_t;
static inline mx_t mx_c(mx_t a, mx_t b) {
  if (b.m > a.m) return b;
  if (a.m > b.m) return a;
  return (b.i < a.i) ? b : a;
}
#pragma omp declare reduction(mxre:mx_t : omp_out = mx_c(omp_out, omp_in))

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
  const int64_t n2 = LEN_2D * LEN_2D;
  mx_t x;
  #pragma omp target teams distribute parallel for reduction(mxre:x) map(to: aa[0:n2]) map(from: bb[0:1])
  for (int64_t k = 0; k < n2; ++k) {
    mx_t t; t.m = aa[k]; t.i = k;
    x = mx_c(x, t);
  }
  const int64_t xi = x.i / LEN_2D;
  const int64_t yi = x.i - xi * LEN_2D;
  bb[0] = x.m + (double)xi + (double)yi;
}
