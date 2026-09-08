/* Parallel scan_affine_decay using OpenMP target offload.
 * Computes y[i] = c[i] * y[i-1] + x[i] for i = 1..LEN_1D-1.
 * Implements a two-pass block algorithm with device offload.
 */

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void scan_affine_decay_fp64(double *restrict y, const double *restrict c,
                            const double *restrict x, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    y[0] = x[0];
    if (LEN_1D == 1) return;

    const int64_t B = 1024; // block size
    const int64_t num_blocks = (LEN_1D - 1 + B - 1) / B;

    // Allocate temporary arrays on host
    double *block_a = (double *)malloc(num_blocks * sizeof(double));
    double *block_b = (double *)malloc(num_blocks * sizeof(double));
    if (!block_a || !block_b) {
        // Fallback to serial computation
        for (int64_t i = 1; i < LEN_1D; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        free(block_a);
        free(block_b);
        return;
    }

    // Phase 1: compute per‑block transformation on the device
    #pragma omp target data map(to: c[0:LEN_1D], x[0:LEN_1D]) \
                            map(tofrom: block_a[0:num_blocks], block_b[0:num_blocks])
    {
        #pragma omp target teams distribute parallel for schedule(static)
        for (int64_t b = 0; b < num_blocks; ++b) {
            int64_t start = 1 + b * B;
            int64_t end = start + B;
            if (end > LEN_1D) end = LEN_1D;
            double a = 1.0;
            double b_acc = 0.0;
            for (int64_t i = start; i < end; ++i) {
                a *= c[i];
                b_acc = c[i] * b_acc + x[i];
            }
            block_a[b] = a;
            block_b[b] = b_acc;
        }
    } // block_a/b now contain results on host

    // Phase 2: compute prefix transformation on host
    double *prefix_a = (double *)malloc(num_blocks * sizeof(double));
    double *prefix_b = (double *)malloc(num_blocks * sizeof(double));
    double *base_y   = (double *)malloc(num_blocks * sizeof(double));
    if (!prefix_a || !prefix_b || !base_y) {
        // Fallback to serial computation
        for (int64_t i = 1; i < LEN_1D; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        free(block_a);
        free(block_b);
        free(prefix_a);
        free(prefix_b);
        free(base_y);
        return;
    }
    prefix_a[0] = 1.0;
    prefix_b[0] = 0.0;
    for (int64_t b = 1; b < num_blocks; ++b) {
        double a_prev = prefix_a[b - 1];
        double b_prev = prefix_b[b - 1];
        double a_block = block_a[b - 1];
        double b_block = block_b[b - 1];
        // compose: (a_block * a_prev, a_block * b_prev + b_block)
        prefix_a[b] = a_block * a_prev;
        prefix_b[b] = a_block * b_prev + b_block;
    }
    // Compute the starting y value for each block
    for (int64_t b = 0; b < num_blocks; ++b) {
        base_y[b] = prefix_a[b] * y[0] + prefix_b[b];
    }

    // Phase 3: compute final y values for each block on the device
    #pragma omp target data map(to: c[0:LEN_1D], x[0:LEN_1D], base_y[0:num_blocks]) \
                            map(tofrom: y[0:LEN_1D])
    {
        #pragma omp target teams distribute parallel for schedule(static)
        for (int64_t b = 0; b < num_blocks; ++b) {
            int64_t start = 1 + b * B;
            int64_t end = start + B;
            if (end > LEN_1D) end = LEN_1D;
            double y_prev = base_y[b];
            for (int64_t i = start; i < end; ++i) {
                double yi = c[i] * y_prev + x[i];
                y[i] = yi;
                y_prev = yi;
            }
        }
    }

    free(block_a);
    free(block_b);
    free(prefix_a);
    free(prefix_b);
    free(base_y);
}
