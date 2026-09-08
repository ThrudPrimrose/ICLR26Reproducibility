/* TSVC tsvc_2 s252: a[i] = b[i]*c[i] + t; t = b[i]*c[i]  (t carries the previous product)
 * => a[i] = b[i]*c[i] + b[i-1]*c[i-1] with b[-1]*c[-1] := 0.0  -> fully parallel.
 * Single device pass; a[0] peeled on the host (bit-exact: s + 0.0).
 * FP_CONTRACT OFF keeps products rounded before the add (matches numpy oracle). */
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D)
{
#pragma STDC FP_CONTRACT OFF
  if (LEN_1D <= 0) return;
  #pragma omp target data map(to: b[0:LEN_1D]) map(to: c[0:LEN_1D]) map(from: a[0:LEN_1D])
  {
    #pragma omp target teams distribute parallel for simd
    for (int64_t i = 1; i < LEN_1D; ++i) {
      a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
    }
  }
  a[0] = b[0] * c[0] + 0.0;
}
