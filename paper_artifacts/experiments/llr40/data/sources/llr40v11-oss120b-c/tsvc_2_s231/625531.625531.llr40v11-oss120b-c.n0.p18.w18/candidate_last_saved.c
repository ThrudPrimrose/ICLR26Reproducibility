#include <stdint.h>

/* Blocked parallel prefix sum.
   - Process columns in blocks to keep a small set of accumulators in registers.
   - For each block, initialise accumulators from the first row (j==0).
   - Then loop over rows, adding bb values to accumulators and writing the
     result to the current row of aa.  This avoids loading the previous aa row
     on every iteration, reducing memory traffic.
   - The outer block loop is parallelised with OpenMP, giving thread-level
     parallelism over independent column blocks.
   - The inner loop over the block is left to the compiler for SIMD.
*/

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t BLOCK = 64; // must be a compile‑time constant
    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < LEN_2D; i0 += BLOCK) {
        int64_t i1 = i0 + BLOCK;
        if (i1 > LEN_2D) i1 = LEN_2D;
        int64_t block_len = i1 - i0;
        double acc[BLOCK];
        // Initialise accumulators from the first row (j == 0)
        for (int64_t b = 0; b < block_len; ++b) {
            acc[b] = aa[i0 + b];
        }
        // Process remaining rows (j = 1 .. LEN_2D-1)
        for (int64_t j = 1; j < LEN_2D; ++j) {
            const double *bb_row = bb + j * LEN_2D + i0;
            double *aa_row = aa + j * LEN_2D + i0;
            for (int64_t b = 0; b < block_len; ++b) {
                acc[b] += bb_row[b];
                aa_row[b] = acc[b];
            }
        }
    }
}
