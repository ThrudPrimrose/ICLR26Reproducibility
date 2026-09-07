#include <stdint.h>

/* KMP substring search: count occurrences of pattern in text.
 * ABI: matches[0] = count; text (N x int64, 0/1), pattern (M x int64, 0/1). */
void kmp_fp64(int64_t *restrict matches, const int64_t *restrict pattern,
              const int64_t *restrict text, const int64_t M, const int64_t N)
{
    /* Build the prefix-failure table. */
    int64_t fail[64];
    fail[0] = 0;
    int64_t k = 0;
    for (int64_t i = 1; i < M; i++) {
        while (k > 0 && pattern[k] != pattern[i]) k = fail[k - 1];
        if (pattern[k] == pattern[i]) k++;
        fail[i] = k;
    }
    /* Scan the text. */
    int64_t count = 0;
    int64_t q = 0;
    for (int64_t i = 0; i < N; i++) {
        int64_t c = text[i];
        while (q > 0 && pattern[q] != c) q = fail[q - 1];
        if (pattern[q] == c) q++;
        if (q == M) {
            count++;
            q = fail[q - 1];
        }
    }
    matches[0] = count;
}
