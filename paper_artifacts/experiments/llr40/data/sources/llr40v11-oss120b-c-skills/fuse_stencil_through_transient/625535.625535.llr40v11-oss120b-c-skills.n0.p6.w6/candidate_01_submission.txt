/* Optimized implementation of fuse_stencil_through_transient_fp64
 * Computes out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])
 * for i = 1 .. LEN_1D-3.
 */
#include <stdint.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    /* Parallelize over i. The loop is independent and vectorizable.
     * Use combined parallel for simd to get both multi-threading and vectorization.
     */
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 1; i < LEN_1D - 2; ++i) {
        double s0 = a[i - 1] + a[i] + a[i + 1];
        double s1 = a[i] + a[i + 1] + a[i + 2];
        out[i] = s0 * s1;
    }
}
