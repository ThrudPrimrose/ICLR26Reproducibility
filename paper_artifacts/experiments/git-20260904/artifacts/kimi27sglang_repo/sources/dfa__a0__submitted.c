#include <stdint.h>
#include <stdlib.h>

void dfa_fp64(int64_t *restrict counts, const int64_t *restrict symbols, const int64_t *restrict trans, const int64_t N, const int64_t NA, const int64_t NS) {
    uint8_t *trans8 = (uint8_t *)malloc((size_t)NS * (size_t)NA);
    for (int64_t i = 0; i < NS * NA; ++i) {
        trans8[i] = (uint8_t)trans[i];
    }
    
    uint8_t state = 0;
    for (int64_t i = 0; i < N; ++i) {
        state = trans8[(size_t)state * (size_t)NA + (size_t)(symbols[i] & 0xFF)];
        counts[state] += 1;
    }
    
    free(trans8);
}
