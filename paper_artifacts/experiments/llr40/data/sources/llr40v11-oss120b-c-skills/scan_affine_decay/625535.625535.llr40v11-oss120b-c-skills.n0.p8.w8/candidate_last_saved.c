// Simple serial implementation of scan_affine_decay kernel.
// Computes the recurrence y[i] = c[i] * y[i-1] + x[i] for i >= 1, with y[0] = x[0].

#include <stdint.h>

void scan_affine_decay_fp64(const double *restrict c, const double *restrict x, double *restrict y, const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        return;
    }
    // Seed: y[0] should already be set to x[0] by the initializer.
    // Overwrite to be safe.
    y[0] = x[0];
    for (int64_t i = 1; i < LEN_1D; ++i) {
        y[i] = c[i] * y[i - 1] + x[i];
    }
}


