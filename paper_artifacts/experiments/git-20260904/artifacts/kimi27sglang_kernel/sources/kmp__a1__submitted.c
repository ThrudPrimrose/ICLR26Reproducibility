#include <stdint.h>
#include <stdlib.h>

void kmp_fp64(int64_t *restrict matches, const int64_t *restrict pattern, const int64_t *restrict text, const int64_t M, const int64_t N, uint8_t *workspace, int64_t workspace_size) {
    (void)workspace;
    (void)workspace_size;

    if (M <= 0 || N <= 0) {
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
    for (int64_t i = 1; i < M; i++) {
        while (k > 0 && pattern[k] != pattern[i]) {
            k = fail[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            k += 1;
        }
        fail[i] = k;
    }

    int64_t count = 0;
    int64_t q = 0;
    for (int64_t i = 0; i < N; i++) {
        while (q > 0 && pattern[q] != text[i]) {
            q = fail[q - 1];
        }
        if (pattern[q] == text[i]) {
            q += 1;
        }
        if (q == M) {
            count += 1;
            q = fail[q - 1];
        }
    }

    matches[0] = count;
    free(fail);
}
