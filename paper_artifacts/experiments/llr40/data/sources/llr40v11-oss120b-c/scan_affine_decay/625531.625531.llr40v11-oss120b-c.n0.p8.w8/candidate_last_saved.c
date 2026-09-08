/* scan_affine_decay kernel implementation (fp64).
 * Computes the recurrence:
 *   y[0] = x[0];
 *   for i = 1..LEN_1D-1:
 *       y[i] = c[i] * y[i-1] + x[i];
 *
 * This version uses a blocked parallel scan to break the data dependence
 * across large arrays and exploit multi-core CPUs via OpenMP.
 *
 * The algorithm consists of three passes:
 *   1) For each block compute the affine transformation (P,Q) such that
 *        y_end = P * y_start + Q
 *      where y_start is the value before the block and y_end the value after.
 *   2) Serially accumulate the block transformations to obtain the initial y
 *      value for each block.
 *   3) In parallel, re‑scan each block using its starting y value to fill the
 *      output array.
 *
 * The block size is chosen to balance sequential work per block and the
 * amount of parallelism. The implementation allocates temporary arrays for the
 * block parameters on the heap.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

/* Use OpenMP for parallelism. The compile command includes -fopenmp. */
#ifdef _OPENMP
#include <omp.h>
#endif

void scan_affine_decay_fp64(double *restrict y, const double *restrict c, const double *restrict x,
                            const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    y[0] = x[0];
    if (LEN_1D == 1) return;

    const int64_t n = LEN_1D;
    const int64_t work_len = n - 1; // number of elements to compute (i = 1..n-1)
    const int64_t BLOCK_SIZE = 256; // tuned for cache, safe for underflow
    const int64_t num_blocks = (work_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double *block_P = (double *)malloc(sizeof(double) * (size_t)num_blocks);
    double *block_Q = (double *)malloc(sizeof(double) * (size_t)num_blocks);
    double *block_start_y = (double *)malloc(sizeof(double) * (size_t)num_blocks);
    if (!block_P || !block_Q || !block_start_y) {
        // fallback to serial
        free(block_P);
        free(block_Q);
        free(block_start_y);
        for (int64_t i = 1; i < n; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
        return;
    }

    // Pass 1: compute (P, Q) for each block in parallel
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t start = 1 + b * BLOCK_SIZE; // inclusive
        int64_t end = start + BLOCK_SIZE;   // exclusive
        if (end > n) end = n;
        double p = 1.0;
        double q = 0.0;
        for (int64_t i = start; i < end; ++i) {
            p = c[i] * p;
            q = c[i] * q + x[i];
        }
        block_P[b] = p;
        block_Q[b] = q;
    }

    // Pass 2: compute starting y for each block
    double y_val = y[0];
    for (int64_t b = 0; b < num_blocks; ++b) {
        block_start_y[b] = y_val;
        y_val = block_P[b] * y_val + block_Q[b];
    }

    // Pass 3: compute y values within each block in parallel
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < num_blocks; ++b) {
        double y_local = block_start_y[b];
        int64_t start = 1 + b * BLOCK_SIZE;
        int64_t end = start + BLOCK_SIZE;
        if (end > n) end = n;
        for (int64_t i = start; i < end; ++i) {
            y_local = c[i] * y_local + x[i];
            y[i] = y_local;
        }
    }

    free(block_P);
    free(block_Q);
    free(block_start_y);
}

