#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

void tsvc_2_s1244_fp64(double * restrict a, const double * restrict b, const double * restrict c, double * restrict d, int64_t LEN_1D, uint8_t * restrict workspace, int64_t workspace_bytes) {
    // Allocate temporary buffer to hold original values of a
    double *old_a = (double *)workspace;
    // If workspace is insufficient, fall back to malloc
    if (workspace == NULL || workspace_bytes < LEN_1D * (int64_t)sizeof(double)) {
        old_a = (double *)malloc((size_t)LEN_1D * sizeof(double));
        if (old_a == NULL) return; // allocation failure, nothing to do
    }
    // Copy original a into old_a
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        old_a[i] = a[i];
    }
    // Main computation: update a[i] and compute d[i] using original a[i+1]
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        double ci = c[i];
        double bi = b[i];
        double ai = bi + ci * ci + bi * bi + ci;
        a[i] = ai;
        d[i] = ai + old_a[i + 1];
    }
    // If we allocated old_a with malloc, free it
    if (workspace == NULL || workspace_bytes < LEN_1D * (int64_t)sizeof(double)) {
        free(old_a);
    }
    // The last element of a (index LEN_1D-1) is left unchanged, as in the reference.
    (void)workspace_bytes; // avoid unused warning
}
