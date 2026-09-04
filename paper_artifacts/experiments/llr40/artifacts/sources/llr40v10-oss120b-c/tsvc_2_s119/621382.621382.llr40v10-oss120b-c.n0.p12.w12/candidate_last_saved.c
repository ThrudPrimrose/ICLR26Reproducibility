#include <stdint.h>
#include <stdlib.h>

/* Parallel-diagonal implementation of TSVC tsvc_2 kernel s119.
 * Computes aa[i,j] = aa[i-1,j-1] + bb[i,j] for i,j = 1..LEN_2D-1.
 * The computation is reorganized over diagonals (i-j = d), which are
 * independent. Each diagonal is processed sequentially, but the outer
 * loop over d is parallelized with OpenMP.
 */

void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    if (LEN_2D <= 1) return;

    const int64_t stride = LEN_2D + 1; // distance between dependent elements in flat array

    // Diagonal index d = i - j ranges from -(LEN_2D-2) .. +(LEN_2D-2).
    // Length of a diagonal = LEN_2D - 1 - |d|.
    #pragma omp parallel for schedule(static)
    for (int64_t d = -(LEN_2D - 2); d <= (LEN_2D - 2); ++d) {
        int64_t len = LEN_2D - 1 - llabs(d);
        if (len <= 0) continue; // safety, should not happen
        // Starting coordinates (i_start, j_start) for this diagonal such that i>=1, j>=1.
        int64_t i_start = (d >= 0) ? 1 + d : 1;
        int64_t j_start = i_start - d; // ensures i_start - j_start == d
        int64_t idx = i_start * LEN_2D + j_start; // flat index of first element
        // Process the diagonal sequentially.
        for (int64_t k = 0; k < len; ++k) {
            aa[idx] = aa[idx - stride] + bb[idx];
            idx += stride;
        }
    }
}
