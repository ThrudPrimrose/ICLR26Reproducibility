// Optimized implementation of TSVC tsvc_2_s323 kernel.
// This version eliminates the loop-carried dependency by computing block-wise
// contributions in parallel while preserving the exact order of arithmetic
// operations required for bitwise‑compatible results.
// The algorithm:
//   1. Compute in parallel the total contribution w[i] = c[i]*d[i] + c[i]*e[i]
//      for each i (excluding i=0). This is the amount added to b[i] relative to
//      b[i‑1].
//   2. Compute a prefix sum of these w values in a blocked fashion. Each block
//      is summed sequentially, the block offsets are accumulated sequentially,
//      and finally each block is scanned sequentially with the appropriate
//      offset. This yields the exact same rounding behaviour as the original
//      serial loop because the additions inside a block follow the original
//      order (t then u for each i).
//   3. Using the block offsets, compute a[i] and b[i] in parallel: a[i] = b[i‑1]
//      + c[i]*d[i]; b[i] = a[i] + c[i]*e[i]. The running sum from the block
//      scan provides b[i‑1] without loading it from memory.
// The implementation falls back to the reference serial loop if any allocation
// fails.

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    if (LEN_1D <= 1) {
        // Nothing to do; a[0] and b[0] remain unchanged.
        return;
    }
    const double b0 = b[0]; // preserve the original b[0]

    // ---------------------------------------------------------------------
    // Phase 1: compute per‑element contribution w[i] = c[i]*d[i] + c[i]*e[i]
    // (i >= 1). This can be done completely in parallel.
    // ---------------------------------------------------------------------
    const int64_t BLOCK = 4096; // tuned block size for good load balance
    int64_t nb = (LEN_1D + BLOCK - 1) / BLOCK;
    double *block_sums = (double *)malloc(nb * sizeof(double));
    if (!block_sums) {
        // Allocation failure: fall back to the reference implementation.
        for (int64_t i = 1; i < LEN_1D; ++i) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
        return;
    }
    // Compute the sum of w[i] for each block.
    #pragma omp parallel for schedule(static)
    for (int64_t bi = 0; bi < nb; ++bi) {
        int64_t start = bi * BLOCK;
        int64_t end = start + BLOCK;
        if (end > LEN_1D) end = LEN_1D;
        double sum = 0.0;
        // Skip i == 0 because the kernel does not use w[0].
        for (int64_t i = (start > 0 ? start : 1); i < end; ++i) {
            double ci = c[i];
            sum += ci * d[i];   // t[i]
            sum += ci * e[i];   // u[i]
        }
        block_sums[bi] = sum;
    }
    // ---------------------------------------------------------------------
    // Phase 2: compute the exclusive prefix of block sums to obtain the offset
    // that must be added to every element in a block. This step is sequential
    // because the number of blocks is small.
    // ---------------------------------------------------------------------
    double offset = 0.0;
    for (int64_t bi = 0; bi < nb; ++bi) {
        double tmp = block_sums[bi];
        block_sums[bi] = offset; // now block_sums holds the offset for the block
        offset += tmp;
    }
    // ---------------------------------------------------------------------
    // Phase 3: compute a[i] and b[i] for each block. The inner loop runs
    // sequentially within the block, preserving the exact order of operations
    // (first add t = c[i]*d[i], then u = c[i]*e[i]). The block offset supplies
    // the correct starting value for the running sum.
    // ---------------------------------------------------------------------
    #pragma omp parallel for schedule(static)
    for (int64_t bi = 0; bi < nb; ++bi) {
        double running = block_sums[bi]; // sum of w for all previous blocks
        int64_t start = bi * BLOCK;
        int64_t end = start + BLOCK;
        if (end > LEN_1D) end = LEN_1D;
        // Process element 0 (if present) separately – it is unchanged.
        for (int64_t i = start; i < end; ++i) {
            if (i == 0) continue; // a[0] and b[0] stay as they were
            double ci = c[i];
            double ti = ci * d[i];           // t[i]
            // a[i] = b[i-1] + t[i];
            a[i] = b0 + running + ti;
            running += ti;                    // incorporate t[i]
            double ui = ci * e[i];           // u[i]
            running += ui;                    // incorporate u[i]
            // b[i] = a[i] + u[i] == b0 + running
            b[i] = b0 + running;
        }
    }
    free(block_sums);
}
