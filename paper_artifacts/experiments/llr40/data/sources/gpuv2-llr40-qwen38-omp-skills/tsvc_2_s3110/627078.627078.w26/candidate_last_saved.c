#include <stdint.h>
#include <omp.h>

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
  const int64_t nn = LEN_2D * LEN_2D;
  double maxv = -1e300;
  int64_t pos = nn;

  #pragma omp target data map(to: aa[0:nn])
  {
    #pragma omp target teams distribute parallel for simd reduction(max: maxv)
    for (int64_t k = 0; k < nn; k++) {
      double v = aa[k];
      if (v > maxv) maxv = v;
    }
    #pragma omp target teams distribute parallel for simd reduction(min: pos)
    for (int64_t k = 0; k < nn; k++) {
      if (aa[k] == maxv && k < pos) pos = k;
    }
  }
  bb[0] = maxv + (double)(pos / LEN_2D) + (double)(pos % LEN_2D);
}
