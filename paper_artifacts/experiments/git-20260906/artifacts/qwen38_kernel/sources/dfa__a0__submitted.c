#include <stdint.h>

/* DFA scan: state = trans[state][symbols[i]]; counts[state] += 1.
 *
 * ABI (cpp_backend/dfa_fp64_binding.json):
 *   void dfa_fp64(int64_t *restrict counts,
 *                 const int64_t *restrict symbols,
 *                 const int64_t *restrict trans,
 *                 const int64_t N, const int64_t NA, const int64_t NS)
 *
 * The state recurrence is a strict dependent chain: one random table lookup
 * per symbol.  The int64 transition table (512KB at NS=NA=256) lives in L2
 * (~12-15 cycles).  Values fit in 8 bits whenever NS <= 256, so we rebuild
 * the table, TRANSPOSED to [symbol][state], as uint8 (64KB).  Two effects:
 *   1. the table shrinks 8x -> ~half the lookups now hit L1;
 *   2. the row base (symbol * rowlen) depends only on the (sequentially
 *      read) symbol stream, so it is computed one step ahead and the
 *      state*NA multiply leaves the critical path (address = base + state).
 * Symbol values are < NA <= 256 on this path, so the row is padded to 256
 * entries and base = sym << 8 (a shift, not a multiply).
 */

static uint8_t dfa_tt2[256 * 256];  /* [sym<<8 | state], sym < NA <= 256 */
static uint8_t dfa_tt8[1u << 20];   /* [sym*NS | state], fallback table  */

void dfa_fp64(int64_t *restrict counts, const int64_t *restrict symbols, const int64_t *restrict trans, const int64_t N, const int64_t NA, const int64_t NS) {
    if (NS <= 0) return;
    if (NS <= 256 && NA <= 256) {
        /* TT2[a*256+s] = trans[s*NA+a];  reads contiguous rows of trans,
         * scatters 8-bit stores (gaps s>=NS are never read). */
        for (int64_t s = 0; s < NS; ++s) {
            const int64_t *row = trans + s * NA;
            for (int64_t a = 0; a < NA; ++a) {
                dfa_tt2[(int64_t)a << 8 | s] = (uint8_t)row[a];
            }
        }
        if (N > 0) {
            uint64_t state = 0;
            uint64_t base = (uint64_t)(uint8_t)symbols[0] << 8;
            int64_t i;
            for (i = 0; i + 1 < N; ++i) {
                state = dfa_tt2[base + state];
                counts[state] += 1;
                base = (uint64_t)(uint8_t)symbols[i + 1] << 8;
            }
            state = dfa_tt2[base + state];
            counts[state] += 1;
        }
    } else if (NS <= 256 && NS * NA <= (1ll << 20)) {
        /* [sym*NS + state] layout, rowlen = NS (multiply, off-critical). */
        for (int64_t s = 0; s < NS; ++s) {
            const int64_t *row = trans + s * NA;
            for (int64_t a = 0; a < NA; ++a) {
                dfa_tt8[a * NS + s] = (uint8_t)row[a];
            }
        }
        if (N > 0) {
            uint64_t state = 0;
            uint64_t base = (uint64_t)symbols[0] * (uint64_t)NS;
            int64_t i;
            for (i = 0; i + 1 < N; ++i) {
                state = dfa_tt8[base + state];
                counts[state] += 1;
                base = (uint64_t)symbols[i + 1] * (uint64_t)NS;
            }
            state = dfa_tt8[base + state];
            counts[state] += 1;
        }
    } else {
        int64_t state = 0;
        for (int64_t i = 0; i < N; ++i) {
            state = trans[state * NA + symbols[i]];
            counts[state] += 1;
        }
    }
}
