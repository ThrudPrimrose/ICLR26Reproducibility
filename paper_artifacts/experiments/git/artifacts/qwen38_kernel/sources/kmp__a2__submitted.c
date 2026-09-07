#include <stdint.h>
#include <stdlib.h>

/* Guess 1: output first (matches), then pattern, text, N, M (like dfa_fp64 convention) */
void kmp_fp64(int64_t *restrict matches, const int64_t *restrict pattern,
              const int64_t *restrict text, int64_t N, int64_t M)
{
    int64_t *fail = (int64_t *)malloc(8 * (M > 0 ? M : 1));
    fail[0] = 0;
    int64_t k = 0;
    for (int64_t i = 1; i < M; i++) {
        while (k > 0 && pattern[k] != pattern[i])
            k = fail[k - 1];
        if (pattern[k] == pattern[i])
            k++;
        fail[i] = k;
    }
    int64_t count = 0;
    int64_t q = 0;
    for (int64_t i = 0; i < N; i++) {
        int64_t c = text[i];
        while (q > 0 && pattern[q] != c)
            q = fail[q - 1];
        if (pattern[q] == c)
            q++;
        if (q == M) {
            count++;
            q = fail[q - 1];
        }
    }
    matches[0] = count;
    free(fail);
}
