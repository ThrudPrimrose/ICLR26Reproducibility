/*
 * Optimized implementation of the TSVC kernel s233.
 * Original reference: /shared/tasks/tsvc_2_s233/tsvc_2_s233_reference.c
 *
 * This version rearranges loops to expose parallelism and vectorization.
 *   - The scan of 'aa' is performed column‑wise; the inner loop over columns is independent
 *     and vectorized with a SIMD pragma.
 *   - The scan of 'bb' is performed row‑wise; the outer loop over rows is parallelized
 *     with OpenMP, each thread handling a distinct row.
 *
 * The function signature and pointer qualifiers exactly match the ABI.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb,
                      const double *restrict cc, const int64_t LEN_2D) {
    /* aa: column‑wise prefix sum (scan down rows).
       Outer loop over rows (j) carries the dependence, inner loop over columns (i) is
       independent and can be vectorized. */
    for (int64_t j = 8; j < LEN_2D; ++j) {
        #pragma omp simd
        for (int64_t i = 8; i < LEN_2D; ++i) {
            aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
        }
    }

    /* bb: row‑wise prefix sum (scan across columns).
       The dependence runs along i, so we parallelize over independent rows (j). */
    #pragma omp parallel for schedule(static)
    for (int64_t j = 8; j < LEN_2D; ++j) {
        for (int64_t i = 8; i < LEN_2D; ++i) {
            bb[j * LEN_2D + i] = bb[j * LEN_2D + (i - 1)] + cc[j * LEN_2D + i];
        }
    }
}

