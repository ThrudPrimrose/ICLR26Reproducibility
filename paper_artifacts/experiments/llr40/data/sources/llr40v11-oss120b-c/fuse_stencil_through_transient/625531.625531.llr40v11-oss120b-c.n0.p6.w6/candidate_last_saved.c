/* Naïve parallel implementation of fuse_stencil_through_transient kernel.
 * Directly computes each output element using the original expression.
 * The implementation is intentionally simple: a parallel for loop over the
 * output indices with unrestricted loads of the source array. This version
 * trades the extra temporary buffer for lower per‑element memory traffic and
 * benefits from OpenMP threading.
 */

#include <stddef.h>
#include <stdint.h>

void fuse_stencil_through_transient_fp64(const double *restrict a,
                                         double *restrict out,
                                         const int64_t LEN_1D) {
    if (LEN_1D <= 3) return;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < LEN_1D - 2; ++i) {
        out[i] = (a[i - 1] + a[i] + a[i + 1]) *
                 (a[i] + a[i + 1] + a[i + 2]);
    }
}
