/* Optimized version of tsvc_2_s1232 kernel.
 * Original reference loop order is:
 *   for (j = 0; j < LEN_2D; ++j)
 *     for (i = j*VLEN; i < LEN_2D; ++i)
 *       aa[i*LEN_2D + j] = bb[i*LEN_2D + j] + cc[i*LEN_2D + j];
 * This version swaps the loops to make the inner loop unit-stride (contiguous)
 * and adds OpenMP parallelization on the outer loop. The inner loop is left to the
 * compiler for SIMD vectorization.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                       const int64_t LEN_2D, const int64_t VLEN) {
    // Parallelize over rows (i) – each iteration writes a distinct row of aa.
    // The inner loop iterates over columns j up to floor(i / VLEN), which is
    // contiguous in memory and therefore vectorizable.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        int64_t j_max = i / VLEN; // floor division; safe because VLEN > 0
        // Ensure we don't exceed the column limit (LEN_2D-1). In practice
        // j_max is always <= LEN_2D-1 for valid inputs, but we guard against
        // pathological VLEN values.
        if (j_max >= LEN_2D) j_max = LEN_2D - 1;
        for (int64_t j = 0; j <= j_max; ++j) {
            aa[i * LEN_2D + j] = bb[i * LEN_2D + j] + cc[i * LEN_2D + j];
        }
    }
}

