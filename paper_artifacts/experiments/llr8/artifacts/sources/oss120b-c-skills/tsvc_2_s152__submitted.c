#include <stdint.h>
#include <omp.h>

// TSVC tsvc_2 kernel s152
// Performs: b[i] = d[i] * e[i]; a[i] = a[i] + b[i] * c[i]
// LEN_1D: length of the 1D arrays
// All arrays are assumed to be of type double.
void tsvc_2_s152_fp64(double *a, double *b, double *c, double *d, double *e, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes) {
    // First loop: compute b = d * e
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double prod = d[i] * e[i];
        b[i] = prod;
        a[i] = a[i] + prod * c[i];
    }
}
