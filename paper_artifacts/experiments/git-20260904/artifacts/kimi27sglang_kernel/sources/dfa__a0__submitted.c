#include <stdint.h>

void dfa_fp64(int64_t *restrict counts, const int64_t *restrict symbols, const int64_t *restrict trans, const int64_t N, const int64_t NA, const int64_t NS, uint8_t *restrict workspace, int64_t workspace_size) {
    int64_t state = 0;
    int64_t trans_size = NS * NA;
    if (trans_size <= 256 * 256) {
        uint8_t stack_u8[256 * 256];
        uint8_t *restrict trans_u8 = (workspace && workspace_size >= trans_size) ? workspace : stack_u8;
        for (int64_t j = 0; j < trans_size; ++j) {
            trans_u8[j] = (uint8_t)trans[j];
        }
        for (int64_t i = 0; i < N; ++i) {
            state = trans_u8[state * NA + symbols[i]];
            counts[state] += 1;
        }
    } else {
        for (int64_t i = 0; i < N; ++i) {
            state = trans[state * NA + symbols[i]];
            counts[state] += 1;
        }
    }
}
