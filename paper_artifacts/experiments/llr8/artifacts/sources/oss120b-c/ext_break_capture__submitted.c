#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

// Vectorized implementation of ext_break_capture with assumed 32‑byte alignment.
// Finds the first index i where a[i] > K and records out_index = i and out_value = a[i].
// Workspace arguments are unused.

void ext_break_capture_fp64(const double * __restrict a,
                            int64_t * __restrict out_index,
                            double * __restrict out_value,
                            int64_t K,
                            int64_t LEN_1D,
                            uint8_t *workspace,
                            int64_t workspace_len) {
    // Initialize sentinel outputs (benchmark guarantees a capture, but keep for safety).
    if (out_index) out_index[0] = -1;
    if (out_value) out_value[0] = -1.0;

    if (LEN_1D <= 0) {
        (void)workspace; (void)workspace_len; return;
    }

    const size_t vec_width = 4; // 4 doubles per AVX2 vector.
    size_t len = (size_t)LEN_1D;
    size_t i = 0;
    size_t vec_end = len / vec_width * vec_width;

    // Assume the input array is 32‑byte aligned for faster loads.
    const double * __restrict a_aligned = (const double *)__builtin_assume_aligned(a, 32);
    __m256d K_vec = _mm256_set1_pd((double)K);

    for (; i < vec_end; i += vec_width) {
        // Load a vector (aligned).
        __m256d a_vec = _mm256_load_pd(a_aligned + i);
        // Compare a_vec > K_vec.
        __m256d cmp = _mm256_cmp_pd(a_vec, K_vec, _CMP_GT_OQ);
        int mask = _mm256_movemask_pd(cmp);
        if (mask == 0) continue;
        // Identify the first lane where the condition is true.
        unsigned tz = __builtin_ctz(mask);
        size_t idx = i + tz;
        out_index[0] = (int64_t)idx;
        out_value[0] = a[idx];
        (void)workspace; (void)workspace_len; return;
    }

    // Process remaining elements scalar.
    for (; i < len; ++i) {
        double v = a[i];
        if (v > (double)K) {
            out_index[0] = (int64_t)i;
            out_value[0] = v;
            (void)workspace; (void)workspace_len; return;
        }
    }
    // No capture (should not happen).
    (void)workspace; (void)workspace_len;
}
