#include <stdint.h>
#include <string.h>

void dfa_fp64(int64_t *restrict counts,
              const int64_t *restrict symbols,
              const int64_t *restrict trans,
              const int64_t N,
              const int64_t NA,
              const int64_t NS) {
    if (N <= 0) {
        return;
    }

    /* Fast path: the reference DFA uses small state sets and alphabets, so
       the transition table fits in L1 cache when stored as bytes.  The values
       are already in [0, NS), so the cast is lossless for NS <= 256. */
    if (NS <= 256 && NA <= 256) {
        uint8_t T[256 * 256];
        const size_t ntrans = (size_t)NS * (size_t)NA;
        for (size_t i = 0; i < ntrans; ++i) {
            T[i] = (uint8_t)trans[i];
        }

        int64_t local[256] = {0};
        uint8_t state = 0;
        const size_t na = (size_t)NA;

        for (int64_t i = 0; i < N; ++i) {
            const uint8_t sym = (uint8_t)symbols[i];
            state = T[(size_t)state * na + sym];
            local[state] += 1;
        }

        for (size_t s = 0; s < (size_t)NS; ++s) {
            counts[s] += local[s];
        }
    } else {
        /* Generic fallback for unusually large state/alphabet sizes. */
        int64_t state = 0;
        for (int64_t i = 0; i < N; ++i) {
            state = trans[state * NA + symbols[i]];
            counts[state] += 1;
        }
    }
}
