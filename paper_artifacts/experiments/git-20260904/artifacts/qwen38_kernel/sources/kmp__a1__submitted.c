#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* KMP substring search: count (overlapping) occurrences of `pattern` (len M)
 * in `text` (len N).  Binary-alphabet fast path: precompute the 8-character
 * transition of the KMP automaton (state in {0..M-1} x 256 groups) and scan
 * with one table lookup per 8 chars.  The scan is a sequential dependency
 * chain, so for large inputs we parallelise via a block "function"
 * composition: each block precomputes (end state, count) for every possible
 * entering state, then the blocks are folded sequentially.  Blocks are sized
 * to stay cache-resident so the M re-reads of a block hit L2/L3. */

void kmp_fp64(int64_t *restrict matches, const int64_t *restrict pattern,
              const int64_t *restrict text, const int64_t M, const int64_t N) {
    if (M <= 0 || N <= 0) { matches[0] = 0; return; }

    int64_t *fail = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    fail[0] = 0;
    int64_t k = 0;
    for (int64_t i = 1; i < M; ++i) {
        while (k > 0 && pattern[k] != pattern[i]) k = fail[k - 1];
        if (pattern[k] == pattern[i]) ++k;
        fail[i] = k;
    }

    int binary = 1;
    for (int64_t i = 0; i < M; ++i)
        if (pattern[i] != 0 && pattern[i] != 1) { binary = 0; break; }

    int64_t count = 0;
    if (!binary) {
        int64_t q = 0;
        for (int64_t i = 0; i < N; ++i) {
            int64_t c = text[i];
            while (q > 0 && pattern[q] != c) q = fail[q - 1];
            if (pattern[q] == c) ++q;
            if (q == M) { ++count; q = fail[M - 1]; }
        }
        matches[0] = count;
        free(fail);
        return;
    }

    /* 8-char transition tables: nxt8/s*256+g = next state, cnt8[...] = matches added. */
    int32_t *nxt8 = (int32_t *)malloc((size_t)M * 256 * sizeof(int32_t));
    int32_t *cnt8 = (int32_t *)malloc((size_t)M * 256 * sizeof(int32_t));
    for (int64_t s = 0; s < M; ++s) {
        for (int64_t g = 0; g < 256; ++g) {
            int64_t q = s, c = 0;
            for (int64_t j = 0; j < 8; ++j) {
                int64_t bit = (g >> j) & 1;
                while (q > 0 && pattern[q] != bit) q = fail[q - 1];
                if (pattern[q] == bit) ++q;
                if (q == M) { ++c; q = fail[M - 1]; }
            }
            nxt8[s * 256 + g] = (int32_t)q;
            cnt8[s * 256 + g] = (int32_t)c;
        }
    }

    int64_t G = N >> 3;          /* full 8-char groups */
    int64_t T = N & 7;           /* tail chars        */
    (void)T;
    int64_t maxt = omp_get_max_threads();

    const int64_t BLOCK_GROUPS = 16384;  /* 128K chars ~ 1 MB, L2/L3 resident */
    int64_t nb = (G + BLOCK_GROUPS - 1) / BLOCK_GROUPS;
    if (nb < 1) nb = 1;

    if (maxt >= 2 && G >= (1 << 16)) {
        int32_t *end_state = (int32_t *)malloc((size_t)nb * M * sizeof(int32_t));
        int32_t *blk_cnt   = (int32_t *)malloc((size_t)nb * M * sizeof(int32_t));
        #pragma omp parallel for schedule(static)
        for (int64_t b = 0; b < nb; ++b) {
            int64_t g0 = b * BLOCK_GROUPS;
            int64_t g1 = g0 + BLOCK_GROUPS;
            if (g1 > G) g1 = G;
            for (int64_t s = 0; s < M; ++s) {
                int64_t q = s, c = 0;
                for (int64_t g = g0; g < g1; ++g) {
                    int64_t base = g << 3;
                    int64_t grp = (int64_t)text[base] | ((int64_t)text[base+1] << 1) |
                                  ((int64_t)text[base+2] << 2) | ((int64_t)text[base+3] << 3) |
                                  ((int64_t)text[base+4] << 4) | ((int64_t)text[base+5] << 5) |
                                  ((int64_t)text[base+6] << 6) | ((int64_t)text[base+7] << 7);
                    c += cnt8[q * 256 + grp];
                    q = nxt8[q * 256 + grp];
                }
                end_state[b * M + s] = (int32_t)q;
                blk_cnt[b * M + s]   = (int32_t)c;
            }
        }
        int64_t q = 0;
        for (int64_t b = 0; b < nb; ++b) {
            count += blk_cnt[b * M + q];
            q = end_state[b * M + q];
        }
        for (int64_t i = G << 3; i < N; ++i) {
            int64_t c = text[i];
            while (q > 0 && pattern[q] != c) q = fail[q - 1];
            if (pattern[q] == c) ++q;
            if (q == M) { ++count; q = fail[M - 1]; }
        }
        free(end_state);
        free(blk_cnt);
    } else {
        int64_t q = 0;
        int64_t i = 0;
        for (; i + 8 <= N; i += 8) {
            int64_t grp = (int64_t)text[i] | ((int64_t)text[i+1] << 1) |
                          ((int64_t)text[i+2] << 2) | ((int64_t)text[i+3] << 3) |
                          ((int64_t)text[i+4] << 4) | ((int64_t)text[i+5] << 5) |
                          ((int64_t)text[i+6] << 6) | ((int64_t)text[i+7] << 7);
            count += cnt8[q * 256 + grp];
            q = nxt8[q * 256 + grp];
        }
        for (; i < N; ++i) {
            int64_t c = text[i];
            while (q > 0 && pattern[q] != c) q = fail[q - 1];
            if (pattern[q] == c) ++q;
            if (q == M) { ++count; q = fail[M - 1]; }
        }
    }

    matches[0] = count;
    free(nxt8);
    free(cnt8);
    free(fail);
}
