#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Faithful C port of the numpy reference (baseline-equivalent). */
void kmp_fp64(int64_t *__restrict__ matches, const int64_t *__restrict__ pattern, const int64_t *__restrict__ text, const int64_t M, const int64_t N) {
    int64_t k;
    int64_t count;
    int64_t q;
    int64_t *fail = (int64_t *)malloc((size_t)(M) * sizeof(int64_t));
    memset(fail, 0, (size_t)(M) * sizeof(int64_t));
    fail[0] = 0;
    k = 0;
    for (int64_t i = 1; i < M; ++i) {
        while (k > 0 && pattern[k] != pattern[i]) {
            k = fail[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            k += 1;
        }
        fail[i] = k;
    }
    count = 0;
    q = 0;
    for (int64_t i = 0; i < N; ++i) {
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
