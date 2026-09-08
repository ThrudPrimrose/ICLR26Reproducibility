/* Optimized version of tsvc_2_s1244_fp64
 * The original loop updates a[i] and then uses a[i+1] (its old value) to compute d[i].
 * This creates a cross-iteration dependence, preventing parallelisation.
 * We preserve the semantics by copying the original a array into a temporary buffer.
 * After the copy we can compute a[i] and d[i] in independent parallel loops.
 */

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b,
                       const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
    // Allocate temporary buffer to hold original a values.
    double *tmp = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (tmp == NULL) {
        // Allocation failure: abort the kernel (no defined error handling).
        return;
    }

    // Copy original a into tmp (preserves the values used for d).
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        tmp[i] = a[i];
    }

    // First loop: compute new a[i] for i < LEN_1D-1.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
    }

    // Second loop: compute d[i] = a[i] + original a[i+1] (from tmp).
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        d[i] = a[i] + tmp[i + 1];
    }

    free(tmp);
}
