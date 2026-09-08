#include <stdint.h>
#include <immintrin.h>

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    const double k = 1.0;
    // Set sentinel values
    out_index[0] = -1;
    out_value[0] = -1.0;

    // Vectorized scan using AVX (process up to 16 doubles per iteration)
    const __m256d vk = _mm256_set1_pd(k);
    int64_t i = 0;
    // Main loop: process 16 elements per iteration (two 8-element blocks)
    for (; i + 15 < LEN_1D; i += 16) {
        // First 8-element block (with prefetch)
        _mm_prefetch((const char*)(a + i + 16), _MM_HINT_T0);
        __m256d v0 = _mm256_loadu_pd(a + i);
        __m256d v1 = _mm256_loadu_pd(a + i + 4);
        __m256d cmp0 = _mm256_cmp_pd(v0, vk, _CMP_GT_OQ);
        __m256d cmp1 = _mm256_cmp_pd(v1, vk, _CMP_GT_OQ);
        int mask0 = _mm256_movemask_pd(cmp0);
        int mask1 = _mm256_movemask_pd(cmp1);
        int mask = mask0 | (mask1 << 4);
        if (mask != 0) {
            int offset = __builtin_ctz(mask);
            int64_t idx = i + offset;
            out_index[0] = idx;
            out_value[0] = a[idx];
            return;
        }
        // Second 8-element block
        __m256d v2 = _mm256_loadu_pd(a + i + 8);
        __m256d v3 = _mm256_loadu_pd(a + i + 12);
        __m256d cmp2 = _mm256_cmp_pd(v2, vk, _CMP_GT_OQ);
        __m256d cmp3 = _mm256_cmp_pd(v3, vk, _CMP_GT_OQ);
        int mask2 = _mm256_movemask_pd(cmp2);
        int mask3 = _mm256_movemask_pd(cmp3);
        int maskB = mask2 | (mask3 << 4);
        if (maskB != 0) {
            int offset = __builtin_ctz(maskB);
            int64_t idx = i + 8 + offset;
            out_index[0] = idx;
            out_value[0] = a[idx];
            return;
        }
    }
    // Process any remaining full 8-element blocks
    for (; i + 7 < LEN_1D; i += 8) {
        __m256d v0 = _mm256_loadu_pd(a + i);
        __m256d v1 = _mm256_loadu_pd(a + i + 4);
        __m256d cmp0 = _mm256_cmp_pd(v0, vk, _CMP_GT_OQ);
        __m256d cmp1 = _mm256_cmp_pd(v1, vk, _CMP_GT_OQ);
        int mask0 = _mm256_movemask_pd(cmp0);
        int mask1 = _mm256_movemask_pd(cmp1);
        int mask = mask0 | (mask1 << 4);
        if (mask != 0) {
            int offset = __builtin_ctz(mask);
            int64_t idx = i + offset;
            out_index[0] = idx;
            out_value[0] = a[idx];
            return;
        }
    }
    // Process any remaining elements (tail < 8)
    for (; i < LEN_1D; ++i) {
        if (a[i] > k) {
            out_index[0] = i;
            out_value[0] = a[i];
            return;
        }
    }
    // If no element greater than k was found, out_index/out_value remain -1.
}

