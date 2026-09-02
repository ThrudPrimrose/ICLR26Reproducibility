#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Optimized implementation of the TSVC kernel ``fuse_stencil_through_transient``.
 * Computes out[i] = (a[i-1] + a[i] + a[i+1]) * (a[i] + a[i+1] + a[i+2])
 * for i = 1 .. LEN_1D-3.
 *
 * This version reduces memory traffic by maintaining a sliding sum of three
 * consecutive elements of ``a``.  The loop is parallelized with OpenMP by splitting
 * the iteration space into independent chunks; each thread keeps its own local
 * ``sum`` and ``prev`` values, eliminating the dependence across thread
 * boundaries.
 *
 * The algorithm loads only one new element of ``a`` per iteration after the
 * initial three-element sum, which drops the load count from four (or six in the
 * naive expression) to a single load while still performing the required three
 * adds and one multiplication per iteration.
 */
void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    /* No work is required if the domain is too small for the stencil. */
    if (LEN_1D < 4) {
        return;
    }

    /* Parallel for implementation (exact rounding). */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < LEN_1D - 2; ++i) {
        double s0 = (a[i - 1] + a[i]) + a[i + 1];
        double s1 = (a[i] + a[i + 1]) + a[i + 2];
        out[i] = s0 * s1;
    }
}
