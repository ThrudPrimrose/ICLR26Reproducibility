/* Optimized implementation of fuse_stencil_through_transient kernel.
 * The reference computes:
 *   out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])
 * for i = 1 .. LEN_1D-3.
 * This version adds OpenMP parallelism and vectorization.
 */
#include <stdint.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    const int64_t start = 1;
    const int64_t end = LEN_1D - 2; // exclusive upper bound
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = start; i < end; ++i) {
        double s1 = a[i - 1] + a[i] + a[i + 1];
        double s2 = a[i] + a[i + 1] + a[i + 2];
        out[i] = s1 * s2;
    }
}
