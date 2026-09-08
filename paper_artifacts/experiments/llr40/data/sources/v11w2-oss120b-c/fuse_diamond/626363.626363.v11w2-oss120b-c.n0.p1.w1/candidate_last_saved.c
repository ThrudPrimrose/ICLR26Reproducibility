/* Optimized version of fuse_diamond_fp64.
 * Computes out[i] = a[i]^4 - 1 for each i in [0, LEN_1D).
 * Uses OpenMP parallel for with SIMD to exploit multi-core and vector units.
 */
#include <stdint.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    /* Parallelize across outer loop, and enable SIMD vectorization.
     * The schedule(static) ensures even distribution of iterations.
     */
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double x = a[i];
        double t = x * x;          // t = a[i]^2
        out[i] = t * t - 1.0;       // out = a[i]^4 - 1
    }
}
