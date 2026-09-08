/* Parallel block prefix sum version using AVX2 intrinsics.
 * The kernel computes aa[j,i] = aa[j-1,i] + bb[j,i] for a LEN_2D x LEN_2D matrix.
 * It is a column-wise prefix sum. Each column is independent, allowing parallelism
 * across blocks of rows. The algorithm proceeds in three phases:
 *   1) Compute the sum of bb rows within each block (block_totals).
 *   2) Compute the starting offset for each block by cumulatively adding block_totals
 *      to the initial row aa[0,*].
 *   3) Re‑scan each block, adding bb to a local per‑column accumulator (offset) and
 *      writing the results back to aa.  AVX2 intrinsics provide vectorised adds.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    if (LEN_2D <= 1) return; // nothing to do

    const int64_t vec_width = 4; // 256‑bit AVX holds 4 doubles
    const int64_t block_rows = 64; // rows per block, tunable for cache

    // Number of blocks covering rows 1..LEN_2D-1 (row 0 is the seed)
    const int64_t nblocks = (LEN_2D - 1 + block_rows - 1) / block_rows;
    // Allocate block totals: one double per column per block
    double *block_totals = (double *)calloc((size_t)nblocks * (size_t)LEN_2D, sizeof(double));
    if (!block_totals) return; // allocation failure – abort silently (will be incorrect)

    /* Phase 1: compute per‑block column sums of bb */
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblocks; ++b) {
        int64_t row_start = 1 + b * block_rows;
        int64_t row_end   = row_start + block_rows;
        if (row_end > LEN_2D) row_end = LEN_2D;
        double *total = block_totals + (size_t)b * (size_t)LEN_2D;
        for (int64_t j = row_start; j < row_end; ++j) {
            const double *brow = bb + (size_t)j * (size_t)LEN_2D;
            int64_t i = 0;
            for (; i + vec_width - 1 < LEN_2D; i += vec_width) {
                __m256d vec_tot = _mm256_loadu_pd(total + i);
                __m256d vec_b   = _mm256_loadu_pd(brow + i);
                vec_tot = _mm256_add_pd(vec_tot, vec_b);
                _mm256_storeu_pd(total + i, vec_tot);
            }
            for (; i < LEN_2D; ++i) {
                total[i] += brow[i];
            }
        }
    }

    /* Phase 2: compute block offsets (starting column values for each block) */
    double *block_offsets = (double *)malloc((size_t)nblocks * (size_t)LEN_2D * sizeof(double));
    if (!block_offsets) { free(block_totals); return; }
    // offset for block 0 is the original first row of aa
    for (int64_t i = 0; i < LEN_2D; ++i) {
        block_offsets[i] = aa[i];
    }
    for (int64_t b = 1; b < nblocks; ++b) {
        double *prev_off = block_offsets + (size_t)(b - 1) * (size_t)LEN_2D;
        double *curr_off = block_offsets + (size_t)b * (size_t)LEN_2D;
        double *prev_tot = block_totals + (size_t)(b - 1) * (size_t)LEN_2D;
        int64_t i = 0;
        for (; i + vec_width - 1 < LEN_2D; i += vec_width) {
            __m256d vec_off = _mm256_loadu_pd(prev_off + i);
            __m256d vec_tot = _mm256_loadu_pd(prev_tot + i);
            vec_off = _mm256_add_pd(vec_off, vec_tot);
            _mm256_storeu_pd(curr_off + i, vec_off);
        }
        for (; i < LEN_2D; ++i) {
            curr_off[i] = prev_off[i] + prev_tot[i];
        }
    }

    /* Phase 3: scan each block, adding bb to a per‑column accumulator and storing to aa */
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblocks; ++b) {
        // Local copy of the offset for this block
        double *offset = (double *)malloc((size_t)LEN_2D * sizeof(double));
        if (!offset) continue; // abort this block on OOM – unlikely
        double *base_off = block_offsets + (size_t)b * (size_t)LEN_2D;
        for (int64_t i = 0; i < LEN_2D; ++i) offset[i] = base_off[i];

        int64_t row_start = 1 + b * block_rows;
        int64_t row_end   = row_start + block_rows;
        if (row_end > LEN_2D) row_end = LEN_2D;
        for (int64_t j = row_start; j < row_end; ++j) {
            const double *brow = bb + (size_t)j * (size_t)LEN_2D;
            double *aout = aa + (size_t)j * (size_t)LEN_2D;
            int64_t i = 0;
            for (; i + vec_width - 1 < LEN_2D; i += vec_width) {
                __m256d vec_off = _mm256_loadu_pd(offset + i);
                __m256d vec_b   = _mm256_loadu_pd(brow + i);
                vec_off = _mm256_add_pd(vec_off, vec_b);
                _mm256_storeu_pd(offset + i, vec_off);
                _mm256_storeu_pd(aout + i, vec_off);
            }
            for (; i < LEN_2D; ++i) {
                offset[i] += brow[i];
                aout[i] = offset[i];
            }
        }
        free(offset);
    }

    free(block_offsets);
    free(block_totals);
}
