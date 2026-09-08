#include <stdint.h>
#include <omp.h>

/* TSVC tsvc_2 s1244 (fp64). Reference semantics, i in [0, LEN_1D-2):
     a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i];
     d[i] = a[i] + a[i+1];
   At the moment d[i] is written, a[i+1] still holds its ORIGINAL (input)
   value: the loop writes a[i] before it ever reads a[i+1] fresh.  So
       d[i] = f(b[i],c[i]) + a_old[i+1],   a[i] = f(b[i],c[i])
   and the two statements live in two independent parallel passes, with the
   d pass running first, while `a` is still the input. */

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c,
                       double *restrict d, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  if (n < 2)
    return;
  const int64_t m = n - 1;

  #pragma omp target data map(to: b[0:n]) map(to: c[0:n]) map(tofrom: a[0:n]) map(from: d[0:n])
  {
    #pragma omp target teams distribute parallel for simd
    for (int64_t i = 0; i < m; i++) {
      d[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i] + a[i + 1];
    }

    #pragma omp target teams distribute parallel for simd
    for (int64_t i = 0; i < m; i++) {
      a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
    }
  }
}
