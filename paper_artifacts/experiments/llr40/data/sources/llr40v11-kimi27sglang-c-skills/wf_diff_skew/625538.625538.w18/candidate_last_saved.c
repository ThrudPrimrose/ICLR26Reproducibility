#include <stdint.h>
#include <omp.h>

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
  const int64_t n = LEN_2D;
  #pragma omp parallel default(none) shared(a, n)
  {
    for (int64_t i = 1; i < n; ++i) {
      #pragma omp for simd schedule(static)
      for (int64_t j = 0; j < n - 1; ++j) {
        a[i * n + j] = a[i * n + j] + a[(i - 1) * n + j] + a[(i - 1) * n + (j + 1)];
      }
    }
  }
}
