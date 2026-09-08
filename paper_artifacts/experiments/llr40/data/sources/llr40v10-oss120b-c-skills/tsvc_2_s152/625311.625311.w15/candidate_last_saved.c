/* Optimized implementation of tsvc_2_s152 kernel.
 * Based on the reference implementation, with OpenMP parallelism and
 * vectorization.
 */

#include <stdint.h>
#include <omp.h>

static inline void s152s_kernel(double *restrict a, const double *restrict b, const double *restrict c, int64_t i) {
    a[i] += b[i] * c[i];
}

void tsvc_2_s152_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    // Parallelize the loop over i. The loop body has no cross-iteration
    // dependencies; each iteration reads/writes distinct elements.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        // Compute temporary b[i]
        b[i] = d[i] * e[i];
        // Update a[i] using the temporary.
        s152s_kernel(a, b, c, i);
    }
}
