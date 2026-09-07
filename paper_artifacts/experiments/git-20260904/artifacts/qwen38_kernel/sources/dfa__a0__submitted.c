#include <stdint.h>
#include <string.h>

/* v0: plain reference translation */
static void dfa_body(int64_t *restrict counts, const int64_t *restrict symbols,
                     const int64_t *restrict trans, const int64_t N,
                     const int64_t NA, const int64_t NS) {
    int64_t state = 0;
    for (int64_t i = 0; i < N; ++i) {
        state = trans[state * NA + symbols[i]];
        counts[state] += 1;
    }
}

void dfa(int64_t *restrict counts, const int64_t *restrict symbols,
         const int64_t *restrict trans, const int64_t N,
         const int64_t NA, const int64_t NS) {
    dfa_body(counts, symbols, trans, N, NA, NS);
}

void dfa_fp64(int64_t *restrict counts, const int64_t *restrict symbols,
              const int64_t *restrict trans, const int64_t N,
              const int64_t NA, const int64_t NS) {
    dfa_body(counts, symbols, trans, N, NA, NS);
}
