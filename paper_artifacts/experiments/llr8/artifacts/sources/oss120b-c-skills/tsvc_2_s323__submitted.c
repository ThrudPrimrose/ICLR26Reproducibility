#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d, const double *restrict e, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    // The first elements a[0] and b[0] remain unchanged.
    // Compute prefix sum of r[i] = c[i] * (d[i] + e[i]) for i >= 1.
    // Use OpenMP inclusive scan to compute b[i] = b[0] + sum_{k=1..i} r[k].
    // Then compute a[i] = b[i] - c[i] * e[i].
    // Simple sequential implementation matching the reference.
    // a[0] and b[0] are left unchanged.
    for (int64_t i = 1; i < LEN_1D; ++i) {
        a[i] = b[i-1] + c[i] * d[i];
        b[i] = a[i] + c[i] * e[i];
    }

}
