#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void dfa_fp64(int64_t *trans,
              int64_t *symbols,
              int64_t *counts,
              int64_t N, int64_t NS, int64_t NA,
              uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    int64_t state = 0;
    for (int64_t i = 0; i < N; ++i) {
        int64_t sym = symbols[i];
        int64_t idx = state * NA + sym;
        if (sym < 0 || sym >= NA) {
            fprintf(stderr, "bad sym at i=%ld: sym=%ld NA=%ld\n", (long)i, (long)sym, (long)NA);
            fflush(stderr);
            return;
        }
        if (idx < 0 || idx >= NS * NA) {
            fprintf(stderr, "bad idx at i=%ld: state=%ld sym=%ld idx=%ld max=%ld\n",
                    (long)i, (long)state, (long)sym, (long)idx, (long)(NS*NA));
            fflush(stderr);
            return;
        }
        state = trans[idx];
        if (state < 0 || state >= NS) {
            fprintf(stderr, "bad state at i=%ld: state=%ld NS=%ld\n", (long)i, (long)state, (long)NS);
            fflush(stderr);
            return;
        }
        counts[state] += 1;
    }
}
