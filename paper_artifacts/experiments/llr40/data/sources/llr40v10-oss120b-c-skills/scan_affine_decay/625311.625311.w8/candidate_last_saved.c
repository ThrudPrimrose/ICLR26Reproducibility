#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>

/*
 * Parallel scan (prefix) for the recurrence:
 *   y[i] = c[i] * y[i-1] + x[i]
 * where y[0] = x[0].
 * This implementation uses a block-wise parallel algorithm.
 * It allocates temporary per-block transformation coefficients (A, B) and
 * per-block start values. The algorithm consists of three phases:
 *   1) Compute, in parallel, for each block its affine transformation
 *      (A, B) such that y[end-1] = A * y[start-1] + B.
 *   2) Compute the start value for each block by a sequential prefix of the
 *      block transformations.
 *   3) Re-scan each block in parallel, using its start value to produce the
 *      final y values.
 *
 * The implementation follows the C-ABI used by the benchmark harness:
 *   void scan_affine_decay_fp64(double *restrict y,
 *                               const double *restrict c,
 *                               const double *restrict x,
 *                               const int64_t LEN_1D);
 * All pointers are marked restrict to aid the optimizer.
 */

void scan_affine_decay_fp64(double *restrict y,
                             const double *restrict c,
                             const double *restrict x,
                             const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    /* Base case: first element has no recurrence term. */
    y[0] = x[0];
    if (LEN_1D == 1) return;

    const int max_threads = omp_get_max_threads();
    // Choose a block size that yields at most one block per thread.
    int64_t block_size = (LEN_1D + max_threads - 1) / max_threads;
    if (block_size < 1) block_size = 1;
    const int nb = (int)((LEN_1D + block_size - 1) / block_size);

    // Allocate temporary workspace for per‑block affine coefficients.
    double *restrict A = (double *restrict)aligned_alloc(64, nb * sizeof(double));
    double *restrict B = (double *restrict)aligned_alloc(64, nb * sizeof(double));
    double *restrict start_y = (double *restrict)aligned_alloc(64, nb * sizeof(double));
    if (!A || !B || !start_y) {
        // Fallback to malloc if aligned_alloc fails.
        if (!A) A = (double *restrict)malloc(nb * sizeof(double));
        if (!B) B = (double *restrict)malloc(nb * sizeof(double));
        if (!start_y) start_y = (double *restrict)malloc(nb * sizeof(double));
        if (!A || !B || !start_y) {
            // Allocation failure – give up silently.
            if (A) free(A);
            if (B) free(B);
            if (start_y) free(start_y);
            return;
        }
    }

    /* Phase 1: compute per‑block affine transform (A, B).
     * For a block covering indices [s, e) (inclusive of s, exclusive of e) we compute:
     *   y[i] = A_i * y[s-1] + B_i   for the last index i = e-1.
     * The recurrence for the coefficients is:
     *   B <- c[i] * B + x[i]
     *   A <- c[i] * A
     */
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < nb; ++b) {
        int64_t s = (int64_t)b * block_size;
        int64_t e = s + block_size;
        if (e > LEN_1D) e = LEN_1D;
        double a = 1.0;
        double bcoeff = 0.0;
        // Skip i = 0 because y[0] is already defined and does not participate in a transform.
        int64_t i_start = (s == 0) ? 1 : s;
        for (int64_t i = i_start; i < e; ++i) {
            bcoeff = c[i] * bcoeff + x[i];
            a = c[i] * a;
        }
        A[b] = a;
        B[b] = bcoeff;
    }

    /* Phase 2: compute the starting y value for each block.
     * start_y[0] is simply y[0]. For subsequent blocks we apply the
     * transformation of the previous block to the previous start value.
     */
    start_y[0] = y[0];
    for (int b = 1; b < nb; ++b) {
        // y at index (block_start - 1) = A[b-1] * start_y[b-1] + B[b-1]
        start_y[b] = A[b-1] * start_y[b-1] + B[b-1];
    }

    /* Phase 3: compute the final y values inside each block.
     * Each block runs independently using its start value.
     */
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < nb; ++b) {
        int64_t s = (int64_t)b * block_size;
        int64_t e = s + block_size;
        if (e > LEN_1D) e = LEN_1D;
        double y_prev;
        if (s == 0) {
            // The first element is already correct; start from i = 1.
            y_prev = y[0];
            for (int64_t i = 1; i < e; ++i) {
                double yi = c[i] * y_prev + x[i];
                y[i] = yi;
                y_prev = yi;
            }
        } else {
            // start_y[b] holds y[s-1]
            y_prev = start_y[b];
            for (int64_t i = s; i < e; ++i) {
                double yi = c[i] * y_prev + x[i];
                y[i] = yi;
                y_prev = yi;
            }
        }
    }

    free(A);
    free(B);
    free(start_y);
}

