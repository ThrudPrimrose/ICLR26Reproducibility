/* Optimized implementation of TSVC tsvc_2_s1232 kernel for double precision.
 * Original reference performs a triangular loop over a 2D matrix.
 * This version restructures loops to enable vectorization of the inner (contiguous) loop
 * and parallel execution of the outer loop using OpenMP.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                       const int64_t LEN_2D, const int64_t VLEN) {
    // Handle the degenerate case VLEN == 0 separately to avoid division by zero.
    if (VLEN == 0) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            #pragma omp simd
            for (int64_t j = 0; j < LEN_2D; ++j) {
                aa[i * LEN_2D + j] = bb[i * LEN_2D + j] + cc[i * LEN_2D + j];
            }
        }
        return;
    }

    // General case: loop over rows (i) outermost, columns (j) innermost.
    // For each row i, the maximum column index is floor(i / VLEN).
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        int64_t jmax = i / VLEN;
        if (jmax >= LEN_2D) jmax = LEN_2D - 1; // safety clamp, usually unnecessary
        #pragma omp simd
        for (int64_t j = 0; j <= jmax; ++j) {
            aa[i * LEN_2D + j] = bb[i * LEN_2D + j] + cc[i * LEN_2D + j];
        }
    }
}
