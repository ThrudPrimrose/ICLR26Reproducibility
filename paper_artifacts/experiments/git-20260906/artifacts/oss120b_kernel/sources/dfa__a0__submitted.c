#include <stdint.h>

/*
 * DFA simulation kernel.
 *
 * trans   : transition table, shape (NS, NA) stored in row-major order.
 * symbols : input symbols array of length N.
 * counts  : histogram of visited state occurrences, length NS.
 * NS      : number of states.
 * NA      : number of symbols (alphabet size).
 * N       : length of the input symbol stream.
 */
void dfa_fp64(const int64_t *restrict trans,
               const int64_t *restrict symbols,
               int64_t *restrict counts,
               int64_t NS,
               int64_t NA,
               int64_t N, uint8_t *restrict workspace, int64_t workspace_bytes) {
    if (NS <= 0 || NA <= 0 || N <= 0) {
        (void)NS; (void)NA; (void)N; // suppress warnings
        return;
    }
    int64_t state = 0;
    for (int64_t i = 0; i < N; ++i) {
        // Transition: new state = trans[current_state, symbol]
        state = trans[state * NA + symbols[i]];
        // Increment visit count for the new state
        counts[state] += 1;
    }
    // NS is unused but part of the required signature.
    (void)NS; (void)workspace; (void)workspace_bytes; // suppress unused parameter warnings
}
