#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#define _GNU_SOURCE
#include <stdint.h>

/* Stream compaction: pack src_v[i] * weight_v[i] for every src_v[i] > 0 and publish the count.
   The input arrays src and weight are of type double, packed is a double output array which is
   assumed to be zero-initialized; out_count is a length‑1 int64 output array. */

#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>

#pragma GCC push_options
#pragma GCC optimize ("no-tree-vectorize")

void compact_threshold_pack_fp64(double *restrict src, double *restrict weight,
                                 double *restrict packed, int64_t *restrict out_count,
                                 int LEN_1D) {
    printf("Entry LEN=%d\n", LEN_1D);
    printf("src=%p weight=%p packed=%p out=%p\n", (void*)src, (void*)weight, (void*)packed, (void*)out_count);
    // Process the arrays in blocks to keep peak memory usage low.
    const int64_t total_len = (int64_t)LEN_1D;
    const int64_t block_size = 1000000; // 1M elements per block
    int64_t n = 0;
    for (int64_t offset = 0; offset < total_len; offset += block_size) {
        int64_t cur_len = total_len - offset;
        if (cur_len > block_size) cur_len = block_size;
        // Process current block.
        for (int64_t i = 0; i < cur_len; ++i) {
            int64_t idx = offset + i;
            double src_val = src[idx];
            if (src_val > 0.0) {
                packed[n] = src_val * weight[idx];
                ++n;
            }
        }
        // Release memory pages of this block; safe because we won't reuse these indices.
        madvise((void*)(src + offset), cur_len * sizeof(double), MADV_DONTNEED);
        madvise((void*)(weight + offset), cur_len * sizeof(double), MADV_DONTNEED);
    }
    out_count[0] = n;
    printf("Done n=%lld\n", (long long)n);
}

#pragma GCC pop_options
