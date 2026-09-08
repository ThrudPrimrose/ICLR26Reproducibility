#include <stdint.h>
#include <omp.h>

/* b[i] = d[i]*e[i]; a[i] += b[i]*c[i] -- no inter-iteration dependence:
 * the only cross-statement flow is within iteration i (b[i] written then read),
 * which a single SIMD lane owns. Fully parallel + vectorizable. */
void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e, const int64_t LEN_1D) {
  #pragma omp parallel for simd schedule(static)
  for (int64_t i = 0; i < LEN_1D; ++i) {
    b[i] = d[i] * e[i];
    a[i] = a[i] + b[i] * c[i];
  }
}
