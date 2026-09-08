/* TSVC tsvc_2_5 kernel fuse_stencil_through_transient, optimized.
 *
 * The fused stencil has no cross-iteration dependence: iteration i reads
 * a[i-1..i+2] and writes out[i] only, with a/out declared restrict.
 * So the whole loop is independent -> thread it, and let the vectorizer
 * handle the unit-stride loads inside each thread.
 */
#include <stdint.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
  #pragma omp parallel for simd schedule(static)
  for (int64_t i = 1; i < LEN_1D - 2; ++i) {
    out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
  }
}
