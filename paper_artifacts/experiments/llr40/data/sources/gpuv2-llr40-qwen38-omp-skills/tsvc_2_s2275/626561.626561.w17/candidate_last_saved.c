#include <stdint.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {

  const int64_t n = LEN_2D;
  const int64_t n2 = n * n;

  #pragma omp target data \
      map(to: b[0:n]) map(to: c[0:n]) map(to: d[0:n]) \
      map(to: bb[0:n2]) map(to: cc[0:n2]) \
      map(tofrom: aa[0:n2]) map(from: a[0:n])
  {
    #pragma omp target
    {
      #pragma clang fp contract(off)
      #pragma omp teams distribute parallel for simd
      for (int64_t k = 0; k < n2; ++k)
        aa[k] = aa[k] + bb[k] * cc[k];
    }

    #pragma omp target
    {
      #pragma clang fp contract(off)
      #pragma omp teams distribute parallel for simd
      for (int64_t i = 0; i < n; ++i)
        a[i] = b[i] + c[i] * d[i];
    }
  }
}
