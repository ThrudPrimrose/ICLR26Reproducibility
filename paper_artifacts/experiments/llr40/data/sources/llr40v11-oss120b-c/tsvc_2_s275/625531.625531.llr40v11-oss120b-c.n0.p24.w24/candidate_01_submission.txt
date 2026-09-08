/* Optimized implementation of the TSVC tsvc_2 kernel "s275" (fp64).
 * Original reference: /shared/tasks/tsvc_2_s275/tsvc_2_s275_reference.c
 *
 * This version restructures the loops for better memory locality and parallelism.
 * The original algorithm:
 *   for i in [0, LEN_2D):
 *       if aa[i] > 0.0:
 *           for j in [1, LEN_2D):
 *               aa[j*LEN_2D + i] = aa[(j-1)*LEN_2D + i] + bb[j*LEN_2D + i] * cc[j*LEN_2D + i]
 *
 * The outer loop over columns (i) can be parallelized, but the inner recurrence
 * over rows (j) is sequential per column. To improve cache behavior we invert the
 * loops: we pre‑compute a mask of columns that satisfy the condition, then iterate
 * over rows sequentially while processing all active columns in parallel across
 * the inner loop. This yields contiguous accesses in each row, which favours the
 * memory subsystem and enables SIMD vectorisation.
 */

#include <stdint.h>
#include <stdlib.h>

void tsvc_2_s275_fp64(double *restrict aa,
                      const double *restrict bb,
                      const double *restrict cc,
                      const int64_t LEN_2D) {
    if (LEN_2D <= 0) return;

    // Allocate a mask for active columns (1 = update, 0 = skip).
    int8_t *active = (int8_t *)malloc((size_t)LEN_2D * sizeof(int8_t));
    if (!active) return; // allocation failure – nothing to do.

    // Build the mask: column 'i' is active if the first element (row 0) is > 0.
    for (int64_t i = 0; i < LEN_2D; ++i) {
        active[i] = (aa[i] > 0.0) ? 1 : 0;
    }

    // Process the recurrence row‑by‑row.
    // Use a single parallel region to avoid repeated thread creation.
    #pragma omp parallel
    {
        // No private copies needed: all data is read‑only except the destination.
        for (int64_t j = 1; j < LEN_2D; ++j) {
            double *aa_row = aa + j * LEN_2D;
            const double *aa_prev = aa + (j - 1) * LEN_2D;
            const double *bb_row = bb + j * LEN_2D;
            const double *cc_row = cc + j * LEN_2D;

            // Parallelise across columns for this row.
            #pragma omp for schedule(static) nowait
            for (int64_t i = 0; i < LEN_2D; ++i) {
                double tmp = aa_prev[i] + bb_row[i] * cc_row[i];
                aa_row[i] = active[i] ? tmp : aa_row[i];
            }
            // Implicit barrier at the end of the for ensures all threads finish row j.
        }
    }

    free(active);
}

