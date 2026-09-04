#include <stdint.h>
#include <omp.h>

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
  const int64_t N = LEN_2D;
  #pragma omp parallel
  {
    for (int64_t j = 0; j < N; j++) {
      const double aj = a[j];
      #pragma omp for
      for (int64_t i = j + 1; i < N; i++) {
        a[i] -= aa[j * N + i] * aj;
      }
    }
  }
}
