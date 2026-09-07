#include <stdint.h>
#include <stdlib.h>

/* DFA scan: state = trans[state*NA + symbols[i]]; counts[state]++.
 * Serial state recurrence. Win: transposed uint8 table so the per-iteration
 * critical path is (sym*NS precompute off-path) + add + byte-load, and the
 * 65KB table fits in L1 where the 2MB int64 table does not. */
void dfa_fp64(int64_t *restrict counts,
              const int64_t *restrict symbols,
              const int64_t *restrict trans,
              int64_t N, int64_t NA, int64_t NS,
              uint8_t *workspace, int64_t workspace_bytes)
{
    if (N <= 0) return;

    /* Fallback for state spaces that do not fit in a byte. */
    if (NS > 256) {
        int64_t state = 0;
        for (int64_t i = 0; i < N; ++i) {
            state = trans[state * NA + symbols[i]];
            counts[state]++;
        }
        return;
    }

    /* Transposed uint8 table: tu8[sym*NS + state] = trans[state*NA + sym]. */
    int64_t tb = NS * NA;
    uint8_t *tu8;
    int mine = 0;
    if (workspace && workspace_bytes >= tb) {
        tu8 = workspace;
    } else {
        tu8 = (uint8_t *)malloc((size_t)tb);
        mine = 1;
    }
    for (int64_t s = 0; s < NS; ++s) {
        const int64_t *tr = trans + s * NA;
        uint8_t *row = tu8 + s;
        for (int64_t a = 0; a < NA; ++a)
            row[a * NS] = (uint8_t)tr[a];
    }

    const int64_t *sym = symbols;
    uint32_t NSu = (uint32_t)NS;
    int64_t state = 0;
    int64_t i = 0;
    for (; i + 7 < N; i += 8) {
        state = tu8[(uint32_t)sym[i + 0] * NSu + state]; counts[state]++;
        state = tu8[(uint32_t)sym[i + 1] * NSu + state]; counts[state]++;
        state = tu8[(uint32_t)sym[i + 2] * NSu + state]; counts[state]++;
        state = tu8[(uint32_t)sym[i + 3] * NSu + state]; counts[state]++;
        state = tu8[(uint32_t)sym[i + 4] * NSu + state]; counts[state]++;
        state = tu8[(uint32_t)sym[i + 5] * NSu + state]; counts[state]++;
        state = tu8[(uint32_t)sym[i + 6] * NSu + state]; counts[state]++;
        state = tu8[(uint32_t)sym[i + 7] * NSu + state]; counts[state]++;
    }
    for (; i < N; ++i) {
        state = tu8[(uint32_t)sym[i] * NSu + state]; counts[state]++;
    }
    if (mine) free(tu8);
}
