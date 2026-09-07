#include <stdint.h>
#include <stdlib.h>

void kmp_fp64(int64_t *matches, int64_t *pattern, int64_t *text,
              int64_t M, int64_t N, uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace; (void)workspace_bytes;
    if (M <= 0 || N <= 0) {
        *matches = 0;
        return;
    }

    if (M <= 64) {
        uint64_t S0 = 0, S1 = 0;
        for (int64_t i = 0; i < M; ++i) {
            uint64_t bit = (uint64_t)1 << (uint64_t)i;
            if (pattern[i] == 0)
                S0 |= bit;
            else
                S1 |= bit;
        }
        const uint64_t match_bit = (uint64_t)1 << (uint64_t)(M - 1);
        uint64_t D = 0;
        int64_t count = 0;
        for (int64_t i = 0; i < N; ++i) {
            uint64_t m = (text[i] == 0) ? S0 : S1;
            D = ((D << 1) | 1) & m;
            if (D & match_bit)
                ++count;
        }
        *matches = count;
        return;
    }

    int64_t *fail = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    if (!fail) {
        *matches = 0;
        return;
    }
    int64_t i, k = 0, q = 0, count = 0;
    fail[0] = 0;
    for (i = 1; i < M; ++i) {
        while (k > 0 && pattern[k] != pattern[i])
            k = fail[k - 1];
        if (pattern[k] == pattern[i])
            ++k;
        fail[i] = k;
    }
    for (i = 0; i < N; ++i) {
        while (q > 0 && pattern[q] != text[i])
            q = fail[q - 1];
        if (pattern[q] == text[i])
            ++q;
        if (q == M) {
            ++count;
            q = fail[q - 1];
        }
    }
    *matches = count;
    free(fail);
}
