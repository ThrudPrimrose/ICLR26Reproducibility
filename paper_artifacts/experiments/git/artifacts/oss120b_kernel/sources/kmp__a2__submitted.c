#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

void kmp_fp64(const int64_t *restrict text, const int64_t *restrict pattern,
               int64_t *restrict matches, int64_t N, int64_t M,
               uint8_t *restrict workspace, int64_t workspace_bytes) {
    // Edge case: empty pattern.
    if (M <= 0) {
        matches[0] = 0;
        return;
    }
    // Pattern longer than text gives zero matches.
    if (M > N) {
        matches[0] = 0;
        return;
    }
    // Guard against overflow when allocating failure table.
    if (M > (int64_t)(SIZE_MAX / sizeof(int64_t))) {
        matches[0] = 0;
        return;
    }
    // Allocate failure (prefix) table. Use malloc; workspace is ignored for simplicity.
    int64_t *fail = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    if (!fail) {
        matches[0] = 0;
        return;
    }
    fail[0] = 0;
    int64_t k = 0;
    for (int64_t i = 1; i < M; ++i) {
        while (k > 0 && pattern[k] != pattern[i]) {
            k = fail[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            ++k;
        }
        fail[i] = k;
    }
    int64_t count = 0;
    int64_t q = 0;
    for (int64_t i = 0; i < N; ++i) {
        while (q > 0 && pattern[q] != text[i]) {
            q = fail[q - 1];
        }
        if (pattern[q] == text[i]) {
            ++q;
        }
        if (q == M) {
            ++count;
            q = fail[q - 1];
        }
    }
    matches[0] = count;
    free(fail);
}

