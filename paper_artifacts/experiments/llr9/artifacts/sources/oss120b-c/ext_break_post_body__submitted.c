#include <stddef.h>
#include <stdint.h>

// Vectorized implementation of ext_break_post_body with early exit handling.
// Performs: a[i] = a[i] + b[i] * c[i] for i = 0..break_idx inclusive,
// where break_idx is the first index such that c[i] > b[i].
// If the condition never holds, the entire array is processed.
// The extra workspace arguments are unused.

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <omp.h>

void ext_break_post_body_fp64(double * __restrict a,
                               double * __restrict b,
                               double * __restrict c,
                               int64_t LEN_1D,
                               uint8_t *workspace,
                               int64_t workspace_len) {
    if (LEN_1D <= 0) {
        (void)workspace; (void)workspace_len;
        return;
    }

    const size_t vec_width = 4; // 4 doubles per AVX2 __m256d
    size_t len = (size_t)LEN_1D;
    size_t i = 0;
    size_t vec_end = len / vec_width * vec_width;

    // Detect the first index where c[i] > b[i].
    size_t break_idx = len; // default: no break.
    // Vectorized detection.
    for (; i < vec_end; i += vec_width) {
        __m256d b_vec = _mm256_loadu_pd(b + i);
        __m256d c_vec = _mm256_loadu_pd(c + i);
        __m256d cmp = _mm256_cmp_pd(c_vec, b_vec, _CMP_GT_OQ);
        int mask = _mm256_movemask_pd(cmp);
        if (mask) {
            unsigned tz = __builtin_ctz(mask);
            break_idx = i + tz;
            goto detection_done;
        }
    }
    // Scalar tail detection.
    for (; i < len; ++i) {
        if (c[i] > b[i]) { break_idx = i; break; }
    }

detection_done:
    // Number of elements to update (inclusive of break index).
    size_t limit = (break_idx == len) ? len : (break_idx + 1);
    // Parallel vectorized body.
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < limit; ++j) {
        a[j] += b[j] * c[j];
    }
    (void)workspace;
    (void)workspace_len;
    return;
}
