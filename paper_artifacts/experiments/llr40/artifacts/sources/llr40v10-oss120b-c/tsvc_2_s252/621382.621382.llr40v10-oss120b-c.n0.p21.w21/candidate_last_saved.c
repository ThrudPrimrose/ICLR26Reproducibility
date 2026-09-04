/* Optimized implementation for tsvc_2_s252 kernel.
 * Computes a[i] = b[i] * c[i] + (i > 0 ? b[i-1] * c[i-1] : 0)
 * This is a 2-point stencil on the elementwise product.
 * The reference implementation uses a sequential recurrence.
 * We accelerate by separating the product into a temporary array and then applying
 * an independent stencil in a second parallel loop, enabling OpenMP parallelism
 * and SIMD vectorization.
 */

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        return;
    }
    // Allocate temporary array for the elementwise product.
    double *s = (double *)malloc((size_t)LEN_1D * sizeof(double));
    if (!s) {
        // Allocation failed – fall back to the original sequential algorithm.
        double t = 0.0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double cur = b[i] * c[i];
            a[i] = cur + t;
            t = cur;
        }
        return;
    }

    // First pass: compute the product b[i] * c[i] in parallel.
    #pragma omp parallel for schedule(static) 
    for (int64_t i = 0; i < LEN_1D; ++i) {
        s[i] = b[i] * c[i];
    }

    // Second pass: apply the 2‑point stencil.
    a[0] = s[0];
    #pragma omp parallel for schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        a[i] = s[i] + s[i-1];
    }

    free(s);
}

