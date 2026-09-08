/* TSVC tsvc_2 kernel "vpvts": a[i] += b[i] * S.
 *
 * Memory-bound saxpy: read a, read b, write a (24 B per element).
 * The loop is embarrassingly parallel (no cross-iteration dependence)
 * and unit-stride, so thread across cores and vectorize the lanes.
 */
#include <stdint.h>
#include <omp.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
  const double s = (double)S;
  #pragma omp parallel for simd schedule(static)
  for (int64_t i = 0; i < LEN_1D; ++i) {
    a[i] += b[i] * s;
  }
}
