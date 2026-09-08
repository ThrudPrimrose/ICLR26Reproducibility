/* Optimized implementation of the fuse_diamond kernel.
 * Computes out[i] = (a[i]^2 + 1)*(a[i]^2 - 1) = a[i]^4 - 1.
 * Uses OpenMP parallel for with SIMD to exploit multi-core and vector units.
 */

#include <stdint.h>
#include <omp.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    // Parallelize across i, static schedule for balanced work.
    // Combine with SIMD for vectorization.
    #pragma omp target teams distribute parallel for simd map(to: a[0:LEN_1D]) map(from: out[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double t = a[i] * a[i];
        // Compute t*t - 1.0, which is equivalent to (t+1)*(t-1) but avoids two adds.
        out[i] = t * t - 1.0;
    }
}
