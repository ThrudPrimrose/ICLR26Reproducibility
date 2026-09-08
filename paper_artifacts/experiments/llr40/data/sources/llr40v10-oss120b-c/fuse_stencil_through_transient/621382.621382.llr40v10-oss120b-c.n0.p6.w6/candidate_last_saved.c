/* Optimized version of fuse_stencil_through_transient_fp64 kernel.
 * Computes out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])
 * for i = 1 .. LEN_1D-3.
 *
 * Uses OpenMP parallel for with SIMD vectorization.
 */
#include <stdint.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    // Guard against small lengths: no work if LEN_1D < 4 (since we need i+2)
    if (LEN_1D < 4) return;
    // Parallelize outer loop; each iteration writes distinct out[i]
    #pragma omp parallel for simd schedule(static) aligned(a:64) aligned(out:64)
    for (int64_t i = 1; i < LEN_1D - 2; ++i) {
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
}
