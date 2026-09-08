#include <stdint.h>
#include <omp.h>

/* Optimized version of tsvc_2_s231_fp64.
 * Original reference performs a column-wise prefix sum across a LEN_2D x LEN_2D
 * matrix stored in column-major order. The dependency is per‑row, so rows are
 * independent. We parallelise across rows and restructure loops for contiguous
 * memory access, enabling vectorisation.
 */

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    // OpenMP parallel region: all threads cooperate for each column.
    // The implicit barrier after the "for" ensures that column j is
    // completely computed before column j+1 starts, preserving the scan
    // dependency across columns.
    #pragma omp parallel
    {
        for (int64_t j = 1; j < LEN_2D; ++j) {
            // Pointers to current and previous column of aa, and current column of bb.
            double *restrict a_cur  = aa + j * LEN_2D;
            double *restrict a_prev = aa + (j - 1) * LEN_2D;
            const double *restrict b_cur = bb + j * LEN_2D;

            // Distribute rows across threads; each iteration works on independent
            // elements, so no race conditions. The barrier at the end of the for
            // (implicit) synchronises before the next column.
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < LEN_2D; ++i) {
                a_cur[i] = a_prev[i] + b_cur[i];
            }
        }
    }
}
