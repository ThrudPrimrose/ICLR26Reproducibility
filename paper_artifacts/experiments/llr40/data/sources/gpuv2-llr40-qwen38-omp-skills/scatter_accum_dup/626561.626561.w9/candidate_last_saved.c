#include <stdint.h>
#include <omp.h>

void scatter_accum_dup_fp64(double *restrict bins, const double *restrict src,
                            const int32_t *restrict ip, const int64_t LEN_1D) {
  const int64_t n = LEN_1D;
  #pragma omp target teams distribute parallel for \
      map(tofrom: bins[0:n]) map(to: src[0:n]) map(to: ip[0:n])
  for (int64_t i = 0; i < n; ++i) {
    #pragma omp atomic
    bins[ip[i]] += src[i];
  }
}
