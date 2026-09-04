#include <stdint.h>
#include <omp.h>

__attribute__((optimize("no-tree-vectorize")))
void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
  if (LEN_1D <= 0) return;

  int nt = omp_get_max_threads();
  double sums[nt];
  double offsets[nt];

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    int64_t chunk = (LEN_1D + nthreads - 1) / nthreads;
    int64_t start = tid * chunk;
    int64_t end = start + chunk;
    if (end > LEN_1D) end = LEN_1D;

    double local_sum = 0.0;
    for (int64_t i = start; i < end; ++i) {
      local_sum += a[i];
    }
    sums[tid] = local_sum;

    #pragma omp barrier
    #pragma omp single
    {
      offsets[0] = 0.0;
      for (int i = 1; i < nthreads; ++i) {
        offsets[i] = offsets[i - 1] + sums[i - 1];
      }
    }
    #pragma omp barrier

    double running = offsets[tid];
    for (int64_t i = start; i < end; ++i) {
      running += a[i];
      b[i] = running;
    }
  }
}
