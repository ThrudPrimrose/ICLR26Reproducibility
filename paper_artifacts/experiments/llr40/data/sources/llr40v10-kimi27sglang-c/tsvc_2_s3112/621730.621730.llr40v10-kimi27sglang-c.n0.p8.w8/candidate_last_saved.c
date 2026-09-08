#include <stdint.h>
#include <omp.h>

__attribute__((optimize("no-tree-vectorize")))
void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;

  double _Alignas(64) sums[2];
  double _Alignas(64) offsets[2];

  #pragma omp parallel num_threads(2)
  {
    const int tid = omp_get_thread_num();
    const int nt = omp_get_num_threads();
    const int64_t half = (LEN_1D + nt - 1) / nt;
    const int64_t start = tid * half;
    int64_t end = start + half;
    if (end > LEN_1D) end = LEN_1D;

    if (tid == 0) {
      double running = 0.0;
      for (int64_t i = start; i < end; ++i) {
        running += a[i];
        b[i] = running;
      }
      sums[0] = running;
    } else {
      double local_sum = 0.0;
      for (int64_t i = start; i < end; ++i) {
        local_sum += a[i];
      }
      sums[1] = local_sum;
    }

    #pragma omp barrier
    #pragma omp single
    {
      offsets[0] = 0.0;
      offsets[1] = sums[0];
    }
    #pragma omp barrier

    if (tid == 1) {
      double running = offsets[1];
      for (int64_t i = start; i < end; ++i) {
        running += a[i];
        b[i] = running;
      }
    }
  }
}
