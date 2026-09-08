/* Block‑wise parallel implementation for the TSVC s323 kernel.
 * The dependence chain is broken by computing the net change contributed by
 * each block on the device, prefix‑summing those block deltas on the host, and
 * then applying the offsets in a second device pass. This avoids large
 * auxiliary arrays and limits data movement, while providing enough parallelism
 * to meet the time budget for the fuzzed input size.
 */

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/* Define a reasonable maximum number of blocks; the actual number may be less
 * if the problem size is small.
 */
#define MAX_BLOCKS 8192

void tsvc_2_s323_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        return;
    }

    // Choose a block size that yields at most MAX_BLOCKS blocks.
    int64_t block = (LEN_1D + MAX_BLOCKS - 2) / MAX_BLOCKS; // ceil((LEN_1D-1)/MAX_BLOCKS)
    if (block < 1) block = 1;
    int64_t nb = (LEN_1D + block - 2) / block; // number of blocks covering i=1..LEN-1

    double *block_delta = (double *)malloc(sizeof(double) * (size_t)nb);
    if (!block_delta) {
        for (int64_t i = 1; i < LEN_1D; ++i) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
        return;
    }

    /* Phase 1: compute the net delta for each block on the device. */
    #pragma omp target data map(to: c[0:LEN_1D], d[0:LEN_1D], e[0:LEN_1D]) map(tofrom: block_delta[0:nb])
    {
        #pragma omp target teams distribute parallel for schedule(static)
        for (int64_t t = 0; t < nb; ++t) {
            int64_t start = t * block + 1;          // first i in this block
            int64_t end   = start + block;         // one past last i
            if (end > LEN_1D) end = LEN_1D;
            double delta = 0.0;
            for (int64_t i = start; i < end; ++i) {
                delta = delta + c[i] * d[i];
                delta = delta + c[i] * e[i];
            }
            block_delta[t] = delta;
        }
    }

    // Compute exclusive prefix of block_delta on the host.
    double *block_offset = (double *)malloc(sizeof(double) * (size_t)nb);
    block_offset[0] = 0.0;
    for (int64_t i = 1; i < nb; ++i) {
        block_offset[i] = block_offset[i - 1] + block_delta[i - 1];
    }

    /* Phase 2: compute final a[] and b[] values block‑wise on the device. */
    #pragma omp target data map(to: c[0:LEN_1D], e[0:LEN_1D], block_offset[0:nb]) map(tofrom: a[0:LEN_1D], b[0:LEN_1D])
    {
        #pragma omp target teams distribute parallel for schedule(static)
        for (int64_t t = 0; t < nb; ++t) {
            int64_t start = t * block + 1;
            int64_t end   = start + block;
            if (end > LEN_1D) end = LEN_1D;
            double b_prev = b[0] + block_offset[t];
            for (int64_t i = start; i < end; ++i) {
                double a_val = b_prev + c[i] * d[i];
                double b_val = a_val + c[i] * e[i];
                a[i] = a_val;
                b[i] = b_val;
                b_prev = b_val; // carry forward within the block
            }
        }
    }

    free(block_delta);
    free(block_offset);
}

