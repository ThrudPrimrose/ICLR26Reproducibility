#include <stdint.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b,
                      const double *restrict bb, const double *restrict c,
                      const int64_t LEN_2D) {
    // Assume 64‑byte alignment for SIMD friendliness.
    double *restrict a_al = (double *)__builtin_assume_aligned(a, 64);
    double *restrict aa_al = (double *)__builtin_assume_aligned(aa, 64);
    const double *restrict b_al = (const double *)__builtin_assume_aligned(b, 64);
    const double *restrict bb_al = (const double *)__builtin_assume_aligned(bb, 64);
    const double *restrict c_al = (const double *)__builtin_assume_aligned(c, 64);

    #pragma omp parallel
    {
        // Update a in parallel (vectorized).
        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            a_al[i] += b_al[i] * c_al[i];
        }
        // Process aa using block-wise approach to reduce memory traffic.
        const int64_t B = 32; // block size (tune for cache)
        // Allocate local buffers (max block size). We'll use fixed-size arrays.
        #pragma omp for schedule(static)
        for (int64_t i0 = 0; i0 < LEN_2D; i0 += B) {
            int64_t i_end = i0 + B;
            if (i_end > LEN_2D) i_end = LEN_2D;
            int64_t blk = i_end - i0;
            // Local buffers for a values and accumulator sums.
            double a_local[32];
            double sum_local[32];
            // Load a values for this block.
                        for (int64_t k = 0; k < blk; ++k) {
                a_local[k] = a_al[i0 + k];
            }
            // Initialize sum with aa column 0 values.
                        for (int64_t k = 0; k < blk; ++k) {
                sum_local[k] = aa_al[i0 + k]; // column 0
            }
            // Iterate over columns j = 1..LEN_2D-1.
            for (int64_t j = 1; j < LEN_2D; ++j) {
                const double *bb_col = bb_al + j * LEN_2D + i0;
                double *aa_cur = aa_al + j * LEN_2D + i0;
                                for (int64_t k = 0; k < blk; ++k) {
                    sum_local[k] += bb_col[k] * a_local[k];
                    aa_cur[k] = sum_local[k];
                }
            }
        }
    }
}
