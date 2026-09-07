// Optimized DFA scan.
//
// The reference is a strict loop-carried recurrence:
//     state = 0
//     for i in 0..N-1: state = trans[state*NA + symbols[i]]; counts[state]++
//
// Two strategies, chosen at runtime from (N, NS, NA, threads):
//   1) Optimized sequential.  The only dependent op is the trans load; the
//      address is state*NA+symbols[i].  When NA is a power of two the multiply
//      becomes a shift, dropping ~2 cycles off the critical path.  The
//      counts[state]++ RMW is fully overlapped (it is not on the state chain).
//   2) Parallel chunk scan (monoid of transition functions).  Works for small
//      NS: each chunk c is summarized by the function G[c]: start-state ->
//      end-state over the chunk's symbols (NS values).  Phases:
//        P1 (parallel): compute G[c][s] for all c,s   -- C*NS*B = N*NS lookups
//        P2 (seq):      chain the start states: start[c]=G[c-1][start[c-1]]
//        P3 (parallel): re-run each chunk from its start state, accumulate a
//                       per-thread histogram, then reduce into counts.
//      The work is N*NS + N, so it only wins when NS is small; the critical
//      path collapses from N to ~2B+C.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static inline int dfa_log2_power2(int64_t NA) {
    if (NA > 0 && (NA & (NA - 1)) == 0) {
        int k = 0; int64_t t = NA;
        while (t > 1) { t >>= 1; k++; }
        return k;
    }
    return -1;
}

static void dfa_seq_shift(int64_t *restrict counts, const int64_t *restrict symbols,
                          const int64_t *restrict trans, int64_t N, int64_t NA, int k) {
    int64_t state = 0;
    if (k >= 0) {
        for (int64_t i = 0; i < N; ++i) { state = trans[(state << k) + symbols[i]]; counts[state] += 1; }
    } else {
        for (int64_t i = 0; i < N; ++i) { state = trans[state * NA + symbols[i]]; counts[state] += 1; }
    }
}

static void dfa_par(int64_t *restrict counts, const int64_t *restrict symbols,
                    const int64_t *restrict trans, int64_t N, int64_t NA, int64_t NS, int nt, int k) {
    int64_t C = 2048;
    if (C > N) C = N;
    if (C > 32 * nt) C = 32 * nt;
    if (C < 1) C = 1;
    int64_t B = (N + C - 1) / C;

    int64_t *G = (int64_t*)malloc((size_t)C * (size_t)NS * 8);
    int64_t *start = (int64_t*)malloc((size_t)C * 8);

    // Phase 1: G[c][s] = end state after chunk c from start state s
    #pragma omp parallel
    {
        int64_t *cur = (int64_t*)malloc((size_t)NS * 8);
        #pragma omp for schedule(static)
        for (int64_t c = 0; c < C; ++c) {
            int64_t i0 = c * B, i1 = i0 + B; if (i1 > N) i1 = N;
            for (int64_t s = 0; s < NS; ++s) cur[s] = s;
            for (int64_t i = i0; i < i1; ++i) {
                int64_t sym = symbols[i];
                for (int64_t s = 0; s < NS; ++s) cur[s] = trans[cur[s]*NA + sym];
            }
            for (int64_t s = 0; s < NS; ++s) G[c*NS + s] = cur[s];
        }
        free(cur);
    }

    // Phase 2: chain start states
    start[0] = 0;
    for (int64_t c = 1; c < C; ++c) start[c] = G[(c-1)*NS + start[c-1]];

    // Phase 3: re-run with per-thread histograms
    #pragma omp parallel
    {
        int64_t *lc = (int64_t*)malloc((size_t)NS * 8);
        memset(lc, 0, (size_t)NS*8);
        #pragma omp for schedule(static)
        for (int64_t c = 0; c < C; ++c) {
            int64_t i0 = c * B, i1 = i0 + B; if (i1 > N) i1 = N;
            int64_t st = start[c];
            for (int64_t i = i0; i < i1; ++i) { st = trans[st*NA + symbols[i]]; lc[st] += 1; }
        }
        #pragma omp critical
        for (int64_t s = 0; s < NS; ++s) counts[s] += lc[s];
        free(lc);
    }
    (void)k;
    free(G); free(start);
}

void dfa_fp64(int64_t *restrict counts, const int64_t *restrict symbols,
              const int64_t *restrict trans, const int64_t N, const int64_t NA, const int64_t NS) {
    int nt = omp_get_max_threads(); if (nt < 1) nt = 1;
    int k = dfa_log2_power2(NA);
    // Use the parallel chunk scan only when it is expected to win:
    // small state space and enough stream length to amortize overhead.
    if (N >= 2000000 && NS <= 3 * (int64_t)nt) {
        dfa_par(counts, symbols, trans, N, NA, NS, nt, k);
    } else {
        dfa_seq_shift(counts, symbols, trans, N, NA, k);
    }
}
