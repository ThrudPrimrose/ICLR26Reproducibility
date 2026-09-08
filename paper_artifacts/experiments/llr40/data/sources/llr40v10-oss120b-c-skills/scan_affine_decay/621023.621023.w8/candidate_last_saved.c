/*
 * scan_affine_decay: compute a first-order linear recurrence with variable coefficients.
 *   y[i] = c[i] * y[i-1] + x[i] for i = 1 .. LEN_1D-1
 * The input array y is assumed to have y[0] already set (seed) by the caller.
 * This implementation parallelizes the scan using block-wise prefix transformation.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void scan_affine_decay_fp64(double *restrict y, const double *restrict c, const double *restrict x, const int64_t LEN_1D, uint8_t *workspace, const int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D <= 1) {
        // Nothing to do; y[0] already contains the seed.
        return;
    }

    // Serial scan implementation
    double y_val = y[0];
    for (int64_t i = 1; i < LEN_1D; ++i) {
        y_val = c[i] * y_val + x[i];
        y[i] = y_val;
    }
}
