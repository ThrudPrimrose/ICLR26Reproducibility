/*
 * Optimized implementation of the TSVC "s2275" microkernel.
 *
 * Original reference implementation (hand‑ported from the TSVC C++ reference):
 *   for i in [0, N):
 *       for j in [0, N):
 *           aa[j*N + i] = aa[j*N + i] + bb[j*N + i] * cc[j*N + i];
 *       a[i] = b[i] + c[i] * d[i];
 *
 * The reference traverses the 2‑D domain column‑major which yields a stride‑N
 * access pattern for the inner loop.  This prevents the compiler from
 * efficiently vectorising the inner loop and also harms cache utilisation.
 *
 * The algorithm is embarrassingly parallel: each element of `aa` is updated
 * independently and the computation of `a[i]` does not depend on any other
 * element.  We therefore reorder the loops to iterate row‑major – this makes
 * the inner loop stride‑1 (contiguous) and allows straight‑forward vectorisation
 * and parallelisation.
 *
 * The implementation below:
 *   * Computes the `a[i]` vector using a simple SIMD loop.
 *   * Updates `aa` row‑wise, employing OpenMP `parallel for` on the outer loop
 *     and an `omp simd` directive on the innermost loop to request vectorisation.
 *   * Uses `restrict` qualifiers on all pointer arguments (as required by the
 *     original prototype) to give the compiler maximal aliasing freedom.
 *
 * The function signature matches the benchmark specification exactly.
 */

#include <stdint.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa,
                       const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc,
                       const double *restrict d, const int64_t LEN_2D)
{
    // Compute a[i] = b[i] + c[i] * d[i] for all i.
    // Independent across i, safe to vectorise.
    #pragma omp simd
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] = b[i] + c[i] * d[i];
    }

    // Update aa with aa[j,i] += bb[j,i] * cc[j,i].
    // Reordered to row‑major (i runs inside) for contiguous memory access.
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < LEN_2D; ++j) {
        // Compute base offset for this row.
        int64_t row_offset = j * LEN_2D;
        const double *bb_row = bb + row_offset;
        const double *cc_row = cc + row_offset;
        double *aa_row = aa + row_offset;
        #pragma omp simd
        for (int64_t i = 0; i < LEN_2D; ++i) {
            aa_row[i] = aa_row[i] + bb_row[i] * cc_row[i];
        }
    }
}
