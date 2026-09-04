#include <stdint.h>
#include <omp.h>

#define BLOCK 4096
#define PAD 8

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 const double *restrict src,
                                 const double *restrict weight,
                                 int64_t LEN_1D,
                                 uint8_t *restrict workspace,
                                 int64_t workspace_size) {
    (void)workspace_size;

    const int64_t seq_threshold = 4096;
    if (LEN_1D < seq_threshold) {
        int64_t n = 0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            if (src[i] > 0.0) {
                packed[n++] = src[i] * weight[i];
            }
        }
        out_count[0] = n;
        return;
    }

    int64_t num_blocks = (LEN_1D + BLOCK - 1) / BLOCK;
    int64_t *block_offsets = (int64_t *)(void *)workspace;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int64_t blocks_per_thread = (num_blocks + nt - 1) / nt;
        int64_t b_start = tid * blocks_per_thread;
        int64_t b_end = b_start + blocks_per_thread;
        if (b_end > num_blocks) b_end = num_blocks;

        /* Count phase */
        for (int64_t b = b_start; b < b_end; ++b) {
            int64_t start = b * BLOCK;
            int64_t end = start + BLOCK;
            if (end > LEN_1D) end = LEN_1D;
            int64_t c = 0;
            for (int64_t i = start; i < end; ++i) {
                if (src[i] > 0.0) {
                    ++c;
                }
            }
            block_offsets[b * PAD] = c;
        }

#pragma omp barrier
#pragma omp single
        {
            int64_t sum = 0;
            for (int64_t b = 0; b < num_blocks; ++b) {
                int64_t c = block_offsets[b * PAD];
                block_offsets[b * PAD] = sum;
                sum += c;
            }
            block_offsets[num_blocks * PAD] = sum;
            out_count[0] = sum;
        }

        /* Pack phase */
        for (int64_t b = b_start; b < b_end; ++b) {
            int64_t start = b * BLOCK;
            int64_t end = start + BLOCK;
            if (end > LEN_1D) end = LEN_1D;
            int64_t out_pos = block_offsets[b * PAD];
            for (int64_t i = start; i < end; ++i) {
                if (src[i] > 0.0) {
                    packed[out_pos++] = src[i] * weight[i];
                }
            }
        }
    }
}
