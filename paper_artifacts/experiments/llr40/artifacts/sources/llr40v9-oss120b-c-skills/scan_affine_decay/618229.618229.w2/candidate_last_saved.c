#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#if 0

/*
 * scan_affine_decay: y[i] = c[i] * y[i-1] + x[i]
 * Input/Output arrays (order as in manifest): y, c, x.
 * LEN_1D is the length of the 1D arrays.
 * y[0] is assumed to be initialized to x[0] by the caller.
 */

void scan_affine_decay_fp64(double *restrict y, const double *restrict c,
                            const double *restrict x, const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        return;
    }
    // Ensure the seed is correctly set.
    y[0] = x[0];
    if (LEN_1D == 1) {
        return;
    }
    // Simple sequential scan to guarantee correctness.
    for (int64_t i = 1; i < LEN_1D; ++i) {
        y[i] = c[i] * y[i-1] + x[i];
    }
    return;
#if 0

    const int64_t BLOCK = 1024; // increased block size for accuracy
    const int64_t num_blocks = (LEN_1D + BLOCK - 1) / BLOCK;

    // Temporary storage for per‑block affine transformation (C, X)
    double *C = (double *)malloc(num_blocks * sizeof(double));
    double *X = (double *)malloc(num_blocks * sizeof(double));
    long double *block_start = (long double *)malloc(num_blocks * sizeof(long double));
    if (!C || !X || !block_start) {
        // allocation failure – fall back to serial execution
        free(C);
        free(X);
        free(block_start);
        // simple serial scan as a safe fallback
        for (int64_t i = 1; i < LEN_1D; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        return;
    }

    /* First pass: compute per‑block transformation (C, X).
       Process each block independently in parallel. */
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t start = b * BLOCK;
        int64_t end = start + BLOCK;
        if (end > LEN_1D) {
            end = LEN_1D;
        }
        // For the first block, skip the seed element at index 0 because it is handled separately.
        int64_t tr_start = (b == 0) ? start + 1 : start;
        // Compute transformation in forward order to correctly accumulate affine coefficients.
        double c_prod = 1.0;
        double x_acc = 0.0;
        for (int64_t i = tr_start; i < end; ++i) {
            x_acc = fma(c[i], x_acc, x[i]);
            c_prod *= c[i];
        }
        // For a block that only contains the seed (possible when LEN_1D == 1), the transformation is identity.
        C[b] = c_prod;
        X[b] = x_acc;
    }

    /* Prefix across blocks to obtain the starting y value for each block. */
    block_start[0] = y[0];
    for (int64_t b = 1; b < num_blocks; ++b) {
        // y_before_block = C_{b-1} * y_before_previous + X_{b-1}
        block_start[b] = fma(C[b - 1], block_start[b - 1], X[b - 1]);
    }

    /* Second pass: compute the actual y values inside each block. */
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t start = b * BLOCK;
        int64_t end = start + BLOCK;
        if (end > LEN_1D) {
            end = LEN_1D;
        }
        double y_prev;
        if (start == 0) {
            // First block already has y[0] set; start from i = 1.
            y_prev = y[0];
            for (int64_t i = 1; i < end; ++i) {
                double y_val = c[i] * y_prev + x[i];
                y[i] = y_val;
                y_prev = y_val;
            }
        } else {
            // For other blocks, start from the value before the block.
            y_prev = (double)block_start[b];
            for (int64_t i = start; i < end; ++i) {
                double y_val = c[i] * y_prev + x[i];
                y[i] = y_val;
                y_prev = y_val;
            }
        }
    }

    free(C);
    free(X);
    free(block_start);
    #endif
}
#endif

#include <stdint.h>

void scan_affine_decay_fp64(double *restrict y, const double *restrict c,
                            const double *restrict x, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    y[0] = x[0];
    for (int64_t i = 1; i < LEN_1D; ++i) {
        y[i] = c[i] * y[i-1] + x[i];
    }
}

