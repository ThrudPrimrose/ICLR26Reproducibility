#include <stdint.h>
#include <immintrin.h>

/* Vectorized implementation of ext_break_capture_fp64.
 * Finds the first element a[i] > 1.0 in the array `a` of length LEN_1D.
 * Writes the index and value to out_index[0] and out_value[0]; if none found,
 * writes -1 and -1.0 (the reference behavior). */
void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    const double k = 1.0;
    // Initialize output to sentinel values.
    out_index[0] = -1;
    out_value[0] = -1.0;

    if (LEN_1D <= 0) {
        return;
    }

    // Vector width (AVX2) = 4 doubles.
    int64_t i = 0;
    // Process in vector chunks.
    const __m256d th = _mm256_set1_pd(k);
    // Process two vectors (8 doubles) per iteration for better throughput.
    for (; i + 8 <= LEN_1D; i += 8) {
        __m256d v0 = _mm256_loadu_pd(a + i);
        __m256d v1 = _mm256_loadu_pd(a + i + 4);
        unsigned int m0 = _mm256_movemask_pd(_mm256_cmp_pd(v0, th, _CMP_GT_OQ));
        unsigned int m1 = _mm256_movemask_pd(_mm256_cmp_pd(v1, th, _CMP_GT_OQ));
        unsigned int mask = m0 | (m1 << 4);
        if (mask) {
            int offset = __builtin_ctz(mask);
            out_index[0] = i + offset;
            out_value[0] = a[i + offset];
            return;
        }
    }
    // Process any remaining full vectors (4 elements each).
    int64_t limit = LEN_1D - (LEN_1D % 4);
    for (; i < limit; i += 4) {
        __m256d v = _mm256_loadu_pd(a + i);
        unsigned int m = _mm256_movemask_pd(_mm256_cmp_pd(v, th, _CMP_GT_OQ));
        if (m) {
            int offset = __builtin_ctz(m);
            out_index[0] = i + offset;
            out_value[0] = a[i + offset];
            return;
        }
    }

    // Process any remaining elements scalar.
    for (; i < LEN_1D; ++i) {
        double v = a[i];
        if (v > k) {
            out_index[0] = i;
            out_value[0] = v;
            return;
        }
    }
    // If we reach here, no element was > k; out_index/out_value already hold the sentinel.
    return;
}
