/* Optimized version of tsvc_2_s1232_fp64
 * Computes aa[i][j] = bb[i][j] + cc[i][j] for i >= j * VLEN.
 * Reordered loops to enable better parallelism and vectorization.
 */

#include <stdint.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D, const int64_t VLEN) {
    // Handle VLEN <= 0 as a full matrix addition (fallback to reference semantics).
    if (VLEN <= 0) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            for (int64_t j = 0; j < LEN_2D; ++j) {
                aa[i * LEN_2D + j] = bb[i * LEN_2D + j] + cc[i * LEN_2D + j];
            }
        }
        return;
    }

    // Parallelize over rows (i). Each iteration processes a contiguous segment of columns (j).
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        // Compute the maximum column index that satisfies i >= j * VLEN.
        // This is equivalent to j <= i / VLEN (integer division).
        int64_t max_j = i / VLEN;
        // Pointers to the start of the i-th row in each matrix.
        double *aa_row = aa + i * LEN_2D;
        const double *bb_row = bb + i * LEN_2D;
        const double *cc_row = cc + i * LEN_2D;
        // Vectorize the inner loop.
        #pragma omp simd
        for (int64_t j = 0; j <= max_j; ++j) {
            aa_row[j] = bb_row[j] + cc_row[j];
        }
    }
}
