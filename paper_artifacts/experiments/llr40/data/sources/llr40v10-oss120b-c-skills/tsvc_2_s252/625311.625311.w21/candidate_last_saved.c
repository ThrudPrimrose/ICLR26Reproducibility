#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    // Allocate temporary workspace for the product of b and c
    // Align to 64 bytes for better SIMD performance.
    double *restrict s = (double *restrict)aligned_alloc(64, (size_t)LEN_1D * sizeof(double));
    if (!s) {
        // Fallback to malloc if aligned_alloc fails (unlikely on modern glibc)
        s = (double *restrict)malloc((size_t)LEN_1D * sizeof(double));
        if (!s) return; // Allocation failure: do nothing
    }
    // First pass: compute elementwise product
    #pragma omp parallel for schedule(static) 
    for (int64_t i = 0; i < LEN_1D; ++i) {
        s[i] = b[i] * c[i];
    }
    // Second pass: compute a[i] = s[i] + s[i-1] with a[0] = s[0]
    if (LEN_1D > 0) {
        a[0] = s[0];
        #pragma omp parallel for schedule(static) 
        for (int64_t i = 1; i < LEN_1D; ++i) {
            a[i] = s[i] + s[i-1];
        }
    }
    free(s);
}

