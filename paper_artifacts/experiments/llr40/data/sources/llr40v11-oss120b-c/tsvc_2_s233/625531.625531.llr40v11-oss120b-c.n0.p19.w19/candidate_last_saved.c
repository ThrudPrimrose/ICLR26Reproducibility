/* Optimized version of the TSVC tsvc_2 kernel ``s233`` (fp64).
 * Original algorithm computes two dependent prefix-sum-like recurrences:
 *   aa[j,i] = aa[j-1,i] + cc[j,i]
 *   bb[j,i] = bb[j,i-1] + cc[j,i]
 * The straightforward implementation iterates over i outer and j inner, leading to
 * strided memory accesses (j*LEN_2D + i).  By swapping the loops we make the inner
 * dimension contiguous in memory, improving cache utilization and allowing the compiler
 * to generate better vector code.  No parallelism can be introduced due to true data
 * dependencies across rows (for aa) and columns (for bb), but the memory layout change
 * yields the main speedup.
 */

#include <stdint.h>

void tsvc_2_s233_fp64(double *restrict aa,
                      double *restrict bb,
                      const double *restrict cc,
                      const int64_t LEN_2D) {
    // Loop over rows (j) outermost, columns (i) innermost for better locality.
    for (int64_t j = 8; j < LEN_2D; ++j) {
        // Compute aa: depends on the value from the previous row (j-1) at the same column.
        // Since we iterate rows sequentially, the dependency is satisfied.
        const double *restrict cc_row = cc + j * LEN_2D;
        const double *restrict aa_prev_row = aa + (j - 1) * LEN_2D;
        double *restrict aa_row = aa + j * LEN_2D;
        for (int64_t i = 8; i < LEN_2D; ++i) {
            aa_row[i] = aa_prev_row[i] + cc_row[i];
        }
        // Compute bb: depends on the previous column (i-1) in the same row.
        // The inner loop proceeds left‑to‑right, satisfying the dependence.
        const double *restrict cc_row_bb = cc_row; // same row as above
        double *restrict bb_row = bb + j * LEN_2D;
        // Note: bb[ j*LEN_2D + (i-1) ] is already computed in this inner loop.
        for (int64_t i = 8; i < LEN_2D; ++i) {
            bb_row[i] = bb_row[i - 1] + cc_row_bb[i];
        }
    }
}

