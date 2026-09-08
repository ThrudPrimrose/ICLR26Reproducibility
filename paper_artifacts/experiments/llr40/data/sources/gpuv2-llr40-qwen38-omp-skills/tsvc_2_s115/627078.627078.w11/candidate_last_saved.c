#include <stdint.h>
#include <omp.h>

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  if (n <= 1) return;
  const int64_t nn = n * n;

  double dot = 0.0;

  #pragma omp target data map(tofrom: a[0:n]) map(to: aa[0:nn])
  {
    #pragma omp target teams distribute parallel for simd reduction(+:dot) collapse(2)
    for (int64_t j = 0; j < n; j++) {
      const int64_t base = j * n;
      for (int64_t i = 0; i < n; i++) {
        dot += aa[base + i] * a[j];
      }
    }

    #pragma omp target teams distribute parallel for simd
    for (int64_t i = 0; i < n; i++) {
      a[i] -= dot * aa[i * n + i];
    }
  }
}
