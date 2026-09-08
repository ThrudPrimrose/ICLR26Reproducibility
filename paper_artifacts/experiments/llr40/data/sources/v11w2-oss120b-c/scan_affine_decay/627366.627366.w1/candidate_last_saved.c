#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/* Optimized implementation of the scan_affine_decay kernel.
 * Argument order expected by the harness:
 *   c   : coefficient array (double*)
 *   x   : input array (double*)
 *   y   : output array (double*)
 *   LEN_1D: length of arrays (int64_t)
 *   workspace: optional workspace buffer (uint8_t*)
 *   workspace_bytes: size of workspace in bytes.
 * The recurrence is: y[i] = c[i] * y[i-1] + x[i] for i >= 1, with y[0] = x[0].
 */

void scan_affine_decay_fp64(double *c,
                            const double *x,
                            double *y,
                            const int64_t LEN_1D,
                            uint8_t *workspace,
                            const int64_t workspace_bytes) {
    if (LEN_1D <= 0) return;
    const int64_t SMALL_LIMIT = 8192;
    if (LEN_1D <= SMALL_LIMIT) {
        y[0] = x[0];
        for (int64_t i = 1; i < LEN_1D; ++i) {
            y[i] = c[i] * y[i - 1] + x[i];
        }
    } else {
        const int64_t BLOCK = 4096;
        const int64_t nblocks = (LEN_1D + BLOCK - 1) / BLOCK;
        // Allocate block transformations (use long double for better range).
        long double *blockA = (long double *)malloc((size_t)nblocks * sizeof(long double));
        long double *blockB = (long double *)malloc((size_t)nblocks * sizeof(long double));
        if (!blockA || !blockB) {
            free(blockA);
            free(blockB);
            // Fallback to sequential.
            y[0] = x[0];
            for (int64_t i = 1; i < LEN_1D; ++i) {
                y[i] = c[i] * y[i - 1] + x[i];
            }
            return;
        }
        // Phase 1: compute per‑block affine transformations.
        #pragma omp parallel for schedule(static)
        for (int64_t b = 0; b < nblocks; ++b) {
            int64_t start = b * BLOCK;
            int64_t end = start + BLOCK;
            if (end > LEN_1D) end = LEN_1D;
            int64_t i_start = (b == 0) ? 1 : start; // skip first element for block 0
            long double A = 1.0L;
            long double B = 0.0L;
            for (int64_t i = i_start; i < end; ++i) {
                long double ci = (long double)c[i];
                long double xi = (long double)x[i];
                B = ci * B + xi;
                A = ci * A;
            }
            blockA[b] = A;
            blockB[b] = B;
        }
        // Phase 2: prefix‑scan block transformations.
        long double *prefixA = (long double *)malloc((size_t)nblocks * sizeof(long double));
        long double *prefixB = (long double *)malloc((size_t)nblocks * sizeof(long double));
        if (!prefixA || !prefixB) {
            free(blockA);
            free(blockB);
            free(prefixA);
            free(prefixB);
            y[0] = x[0];
            for (int64_t i = 1; i < LEN_1D; ++i) {
                y[i] = c[i] * y[i - 1] + x[i];
            }
            return;
        }
        prefixA[0] = 1.0L;
        prefixB[0] = 0.0L;
        for (int64_t b = 1; b < nblocks; ++b) {
            long double Aprev = blockA[b - 1];
            long double Bprev = blockB[b - 1];
            prefixA[b] = Aprev * prefixA[b - 1];
            prefixB[b] = Aprev * prefixB[b - 1] + Bprev;
        }
        // Phase 3: compute final y values.
        #pragma omp parallel for schedule(static)
        for (int64_t b = 0; b < nblocks; ++b) {
            int64_t start = b * BLOCK;
            int64_t end = start + BLOCK;
            if (end > LEN_1D) end = LEN_1D;
            if (b == 0) {
                // Block 0: compute sequentially from the seed.
                y[0] = x[0];
                for (int64_t i = 1; i < end; ++i) {
                    y[i] = c[i] * y[i - 1] + x[i];
                }
            } else {
                // Compute y value just before this block.
                long double y_before = prefixA[b] * (long double)x[0] + prefixB[b]; // x[0] == y[0]
                long double y_curr = y_before;
                for (int64_t i = start; i < end; ++i) {
                    y_curr = (long double)c[i] * y_curr + (long double)x[i];
                    y[i] = (double)y_curr;
                }
            }
        }
        free(blockA);
        free(blockB);
        free(prefixA);
        free(prefixB);
    }
    // Copy result to workspace if provided.
    if (workspace != NULL) {
        int64_t max_copy = workspace_bytes / (int64_t)sizeof(double);
        if (max_copy > LEN_1D) max_copy = LEN_1D;
        double *ws = (double *)workspace;
        for (int64_t i = 0; i < max_copy; ++i) {
            ws[i] = y[i];
        }
    }
}
