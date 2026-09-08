#include <stddef.h>
#include <stdint.h>
#include <omp.h>

/*
 * Optimized implementation of the TSVC tsvc_2 ``s255`` kernel.
 * Computes a circular 3‑point average of array `b` into array `a`.
 * Original reference uses a scalar recurrence with variables `x` and `y`.
 * This version removes the loop‑carried dependency by expressing the
 * recurrence directly as a sum of three neighbouring elements (with
 * wrap‑around). The first two elements are handled as a special case
 * and the remaining loop (i >= 2) is both SIMD‑vectorizable and
 * thread‑parallel using OpenMP.
 */

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    const double factor = 0.333; /* multiply by 1/3 */
    int64_t n = LEN_1D;
    if (n <= 0) {
        return;
    }
    if (n == 1) {
        /* With a single element the circular window consists of three copies
         * of that element, yielding the original value. */
        a[0] = b[0];
        return;
    }
    if (n == 2) {
        // a[0] = (b[0] + b[1] + b[0]) * factor;
        // a[1] = (b[1] + b[0] + b[1]) * factor;
        a[0] = (b[0] + b[1] + b[0]) * factor;
        a[1] = (b[1] + b[0] + b[1]) * factor;
        return;
    }
    // n >= 3
    // Handle the first two elements with wrap‑around indices.
    a[0] = (b[0] + b[n-1] + b[n-2]) * factor;
    a[1] = (b[1] + b[0]   + b[n-1]) * factor;

    // Vectorizable and parallel part for i >= 2.
    #pragma omp parallel for simd schedule(static) aligned(a,b:64)
    for (int64_t i = 2; i < n; ++i) {
        a[i] = (b[i] + b[i-1] + b[i-2]) * factor;
    }
}

