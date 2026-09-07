#include <stdint.h>
void dfa_fp64(int64_t *restrict counts, int64_t *restrict symbols, int64_t *restrict trans,
              int64_t N, int64_t NA, int64_t NS, uint8_t *ws, int64_t wssize) {
    int64_t state = 0;
    for (int64_t i = 0; i < N; i++) {
        state = trans[state * NA + symbols[i]];
        counts[state] += 1;
    }
    (void)NS; (void)ws; (void)wssize;
}
