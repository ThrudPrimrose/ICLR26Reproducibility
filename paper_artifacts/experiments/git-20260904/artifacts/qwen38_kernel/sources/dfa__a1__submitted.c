#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

/* Entry: counts[NS], symbols[N], trans[NS*NA row-major], N, NA, NS, workspace, ws_bytes
 * Reference: state=0; for i<N: state=trans[state][symbols[i]]; counts[state]++;
 *
 * Two-phase segmented scan with compact (union-find) block transition functions.
 * Random DFAs collapse each block function to ~1 distinct value within a few dozen
 * transitions, so phase 1 costs ~O(N) not O(N*NS). Phases run windows of W blocks
 * in parallel per chunk to hide dependent-load latency.
 */

typedef struct {
    uint8_t  *val;      /* B*NS region for this block: function values (valid on roots) */
    uint8_t  *par;      /* B*NS region: union-find parents over q */
    uint8_t   srep[256], snv[256];
    uint32_t  stamp[256];
    uint8_t   owner[256];
    int64_t   m;
    uint32_t  cur;
    uint8_t  *cref;     /* when m==1: &val[root] */
} P1Slot;

static inline void p1_step(P1Slot *st, const uint8_t *col)
{
    if (st->m == 1) {
        *st->cref = col[*st->cref];
        st->cur++;
    } else {
        int64_t m = st->m, m2 = 0;
        uint8_t *srep = st->srep, *snv = st->snv;
        uint32_t cur = st->cur;
        uint32_t *stamp = st->stamp;
        uint8_t *owner = st->owner;
        uint8_t *val = st->val;
        uint8_t *par = st->par;
        for (int64_t j = 0; j < m; j++) {
            uint8_t r = srep[j];
            uint8_t nv = col[val[r]];
            if (stamp[nv] != cur) {
                stamp[nv] = cur;
                owner[nv] = r;
                snv[m2] = nv;
                srep[m2] = r;
                m2++;
            } else {
                par[r] = owner[nv];
            }
        }
        for (int64_t j = 0; j < m2; j++) val[srep[j]] = snv[j];
        st->m = m2;
        st->cur = cur + 1;
        if (m2 == 1) st->cref = val + srep[0];
    }
}

static inline void p1_init(P1Slot *st, uint8_t *val, uint8_t *par, int64_t NS)
{
    st->val = val;
    st->par = par;
    for (int64_t q = 0; q < NS; q++) {
        val[q] = (uint8_t)q;
        par[q] = (uint8_t)q;
        st->srep[q] = (uint8_t)q;
        st->stamp[q] = 0;
    }
    st->m = NS;
    st->cur = 1;
    st->cref = (NS == 1) ? val : NULL;
}

void dfa_fp64(int64_t *counts, int64_t *symbols, int64_t *trans,
              int64_t N, int64_t NA, int64_t NS,
              uint8_t *workspace, int64_t workspace_bytes)
{
    /* ---------- naive fallback ---------- */
    if (NS > 256 || N <= 0) {
        int64_t state = 0;
        for (int64_t i = 0; i < N; i++) {
            state = trans[state * NA + symbols[i]];
            counts[state] += 1;
        }
        return;
    }

    const int64_t BMAX = 4096;
    const int64_t W = 8;
    const int64_t P = omp_get_max_threads();

    int64_t B = N < BMAX ? N : BMAX;
    int64_t L = (N + B - 1) / B;
    B = (N + L - 1) / L;   /* drop any fully-empty blocks: B*L >= N and each block has >= 1 symbol */

    int64_t need = NA * NS + 2 * B * NS + B + 2048 * NS;
    uint8_t *ws = workspace;
    int own_ws = 0;
    if (!ws || workspace_bytes < need) {
        ws = (uint8_t *)malloc(need);
        own_ws = 1;
        if (!ws) {
            int64_t state = 0;
            for (int64_t i = 0; i < N; i++) {
                state = trans[state * NA + symbols[i]];
                counts[state] += 1;
            }
            return;
        }
    }

    /* ---------- byte table: Tb[s*NS + q] = trans[q*NA + s] ---------- */
    uint8_t *Tb = ws;
    for (int64_t s = 0; s < NA; s++)
        for (int64_t q = 0; q < NS; q++)
            Tb[s * NS + q] = (uint8_t)trans[q * NA + s];

    uint8_t *bval    = Tb + NA * NS;
    uint8_t *bpar    = bval + B * NS;
    uint8_t *starts  = bpar + B * NS;
    int64_t *histws  = (int64_t *)(starts + B);

    #pragma omp parallel num_threads(P)
    {
        int64_t tid = omp_get_thread_num();

        /* ---------- Phase 1: per-block transition functions ---------- */
        #pragma omp for schedule(static)
        for (int64_t base = 0; base < B; base += W) {
            int64_t nw = B - base < W ? B - base : W;
            int64_t gi0 = base * L;
            int64_t lastLen = L;
            if (base + nw == B) lastLen = N - (B - 1) * L;

            P1Slot st[8];
            for (int64_t w = 0; w < nw; w++)
                p1_init(&st[w], bval + (base + w) * NS, bpar + (base + w) * NS, NS);

            int64_t i;
            for (i = 0; i < lastLen; i++) {
                __builtin_prefetch(symbols + gi0 + i + 256, 0, 3);
                for (int64_t w = 0; w < nw; w++)
                    p1_step(&st[w], Tb + (int64_t)symbols[gi0 + w * L + i] * NS);
            }
            if (lastLen < L)
                for (; i < L; i++)
                    for (int64_t w = 0; w < nw - 1; w++)
                        p1_step(&st[w], Tb + (int64_t)symbols[gi0 + w * L + i] * NS);
        }

        /* ---------- Phase 2: chain the block starts (serial) ---------- */
        #pragma omp single
        {
            int64_t state = 0;
            for (int64_t b = 0; b < B; b++) {
                starts[b] = (uint8_t)state;
                uint8_t *par = bpar + b * NS;
                uint8_t *val = bval + b * NS;
                int64_t s = state;
                while (par[s] != s) s = par[s];
                state = val[s];
            }
        }
        #pragma omp barrier

        /* ---------- Phase 3: replay trajectories, local histograms ---------- */
        int64_t *hist = histws + tid * NS;
        for (int64_t q = 0; q < NS; q++) hist[q] = 0;

        #pragma omp for schedule(static)
        for (int64_t base = 0; base < B; base += W) {
            int64_t nw = B - base < W ? B - base : W;
            int64_t gi0 = base * L;
            int64_t lastLen = L;
            if (base + nw == B) lastLen = N - (B - 1) * L;

            int64_t s[8];
            for (int64_t w = 0; w < nw; w++) s[w] = starts[base + w];

            int64_t i;
            for (i = 0; i < lastLen; i++) {
                __builtin_prefetch(symbols + gi0 + i + 256, 0, 3);
                for (int64_t w = 0; w < nw; w++) {
                    s[w] = Tb[(int64_t)symbols[gi0 + w * L + i] * NS + s[w]];
                    hist[s[w]] += 1;
                }
            }
            if (lastLen < L)
                for (; i < L; i++)
                    for (int64_t w = 0; w < nw - 1; w++) {
                        s[w] = Tb[(int64_t)symbols[gi0 + w * L + i] * NS + s[w]];
                        hist[s[w]] += 1;
                    }
        }

        /* ---------- Phase 4: reduce histograms ---------- */
        #pragma omp for schedule(static)
        for (int64_t q = 0; q < NS; q++) {
            int64_t acc = 0;
            for (int64_t p = 0; p < P; p++) acc += histws[p * NS + q];
            counts[q] += acc;
        }
    }

    if (own_ws) free(ws);
}
