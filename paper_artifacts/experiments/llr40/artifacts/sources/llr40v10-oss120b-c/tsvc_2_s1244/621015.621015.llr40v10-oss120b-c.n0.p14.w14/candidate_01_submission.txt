/* Optimized implementation of TSVC kernel s1244 (double precision).
   Function semantics match the reference implementation:
       a[i] = b[i] + c[i] + c[i]*c[i] + b[i]*b[i]
       d[i] = a[i] + original_a[i+1]
   for i = 0 .. LEN_1D-2.

   To preserve the "original_a[i+1]" value we first copy the input array `a`
   into a temporary buffer. The three loops are parallelized with OpenMP; the
   compute loops are straightforward enough for the compiler to auto‑vectorize.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b,
                       const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        return; // nothing to do
    }

    /* Allocate a buffer to hold the original contents of `a`. */
    double *restrict a_orig = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (!a_orig) {
        /* On allocation failure fall back to the reference implementation. */
        for (int64_t i = 0; i < LEN_1D - 1; ++i) {
            a[i] = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
            d[i] = a[i] + a[i + 1];
        }
        return;
    }

    /* 1) Copy the original `a` values – needed for the d[i] computation. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a_orig[i] = a[i];
    }

    /* 2) Compute the new values of `a` (skip the last element, which stays unchanged). */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        double bi = b[i];
        double ci = c[i];
        a[i] = bi + ci + ci * ci + bi * bi;
    }

    /* 3) Compute `d` using the newly computed `a` and the saved original values. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        d[i] = a[i] + a_orig[i + 1];
    }

    free(a_orig);
}

