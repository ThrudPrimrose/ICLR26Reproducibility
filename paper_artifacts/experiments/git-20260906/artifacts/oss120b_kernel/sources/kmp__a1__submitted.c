#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void kmp_fp64(int64_t *text, int64_t *pattern, int64_t *matches, int64_t N, int64_t M, uint8_t *workspace, int64_t workspace_bytes) {
    printf("debug N=%lld M=%lld\n", (long long)N, (long long)M);
    if (M <= 0) {
        matches[0] = 0;
        return;
    }
    // Allocate prefix table (failure function)
    // If pattern longer than text, no matches possible.
    if (M > N) {
        matches[0] = 0;
        return;
    }
    // Guard against integer overflow in allocation size.
    if (M > (int64_t)(SIZE_MAX / sizeof(int64_t))) {
        matches[0] = 0;
        return;
    }
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
