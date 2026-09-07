// Optimized DFA scan.
//
// The walk `state = trans[state][symbols[i]]` is a strict loop-carried
// recurrence, so no amount of thread/vector parallelism can break the state
// chain; the only winning moves are on the dependent load itself:
//
//  1. The table values are < NS <= 256, so build a column-major *byte* table
//     tb[s*NS + x] = trans[x*NA + s] (NS*NA bytes: 64KB at 256x256 instead of
//     512KB). The dependent access becomes tb[b + x] with the row base b
//     computed one step ahead: the per-step chain is a single [base+reg]
//     load, and a 64KB table yields a far higher L1 hit rate than 512KB.
//  2. Software-prefetch the 4 cache lines of column symbols[i+D] into L1
//     (PREFETCHT0) a few steps before the walk reaches it, so the dependent
//     load hits L1 (~4c) instead of L2 (~12c) on Zen4.
//  3. 4x unroll with all bases precomputed, amortizing loop overhead.
//
// General NS > 256 (or oversized table) falls back to the plain row-major
// int64 walk, which is still faster than the naive baseline.
#include <stdint.h>
#include <stddef.h>

static inline void pref_col_n(const uint8_t *tb, int64_t c, int nlines)
{
    __builtin_prefetch(tb + c + 0, 0, 3);
    if (nlines > 1) __builtin_prefetch(tb + c + 64, 0, 3);
    if (nlines > 2) __builtin_prefetch(tb + c + 128, 0, 3);
    if (nlines > 3) __builtin_prefetch(tb + c + 192, 0, 3);
}
static inline int col_lines(int64_t ns) { return (int)(((ns) + 63) >> 6); }

static void dfa_walk_int64(int64_t *restrict counts, const int64_t *restrict symbols,
                           const int64_t *restrict trans, const int64_t N, const int64_t NA)
{
    int64_t st = 0;
    for (int64_t i = 0; i < N; i++) {
        st = trans[st * NA + symbols[i]];
        counts[st] += 1;
    }
}

static void dfa_walk_byte(int64_t *restrict counts, const int64_t *restrict symbols,
                          uint8_t *restrict tb, const int64_t N, const int64_t NA, const int64_t NS)
{
    (void)NA;
    if (N < 4096) {
        int64_t x = 0;
        for (int64_t i = 0; i < N; i++) {
            uint8_t v = tb[symbols[i] * NS + x];
            counts[v] += 1;
            x = v;
        }
        return;
    }

    const int64_t *sym = symbols;
    const int nlines = col_lines(NS);
    int64_t x = 0;
    int64_t i = 0;
    /* prime: run the first 3 steps while issuing their prefetches */
    for (; i < 3; i++) {
        pref_col_n(tb, sym[i + 3] * NS, nlines);
        uint8_t v = tb[sym[i] * NS + x];
        counts[v] += 1;
        x = v;
    }
    /* main: 4x unrolled; columns i+3..i+6 are prefetched while steps
     * i..i+3 run. Loop invariant i + 7 < N keeps every symbol read in
     * bounds; the tail (<= 10 steps) runs without prefetches. */
    for (; i + 7 < N; i += 4) {
        pref_col_n(tb, sym[i + 3] * NS, nlines);
        pref_col_n(tb, sym[i + 4] * NS, nlines);
        pref_col_n(tb, sym[i + 5] * NS, nlines);
        pref_col_n(tb, sym[i + 6] * NS, nlines);
        const uint8_t *b0 = tb + sym[i + 0] * NS;
        const uint8_t *b1 = tb + sym[i + 1] * NS;
        const uint8_t *b2 = tb + sym[i + 2] * NS;
        const uint8_t *b3 = tb + sym[i + 3] * NS;
        uint8_t v;
        v = b0[x]; counts[v] += 1; x = v;
        v = b1[x]; counts[v] += 1; x = v;
        v = b2[x]; counts[v] += 1; x = v;
        v = b3[x]; counts[v] += 1; x = v;
    }
    for (; i < N; i++) {
        uint8_t v = tb[sym[i] * NS + x];
        counts[v] += 1;
        x = v;
    }
#undef PREF_COL
}

void dfa_fp64(int64_t *restrict counts, const int64_t *restrict symbols,
              const int64_t *restrict trans, const int64_t N, const int64_t NA,
              const int64_t NS, uint8_t *restrict workspace, const int64_t workspace_size)
{
    if (N <= 0 || NS <= 0 || NA <= 0) return;

    if (NS <= 256) {
        int64_t need = NS * NA;
        uint8_t *tb = NULL;
        if (workspace && workspace_size >= need) {
            tb = workspace;
        } else {
            static uint8_t st_buf[1 << 20];
            if (need <= (int64_t)(1 << 20)) tb = st_buf;
        }
        if (tb) {
            /* column-major byte table: tb[s*NS + x] = trans[x*NA + s] */
            for (int64_t x = 0; x < NS; x++) {
                const int64_t *row = trans + x * NA;
                for (int64_t s = 0; s < NA; s++)
                    tb[s * NS + x] = (uint8_t)row[s];
            }
            dfa_walk_byte(counts, symbols, tb, N, NA, NS);
            return;
        }
    }
    dfa_walk_int64(counts, symbols, trans, N, NA);
}
