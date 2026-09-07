/* DFA (deterministic finite automaton) traversal kernel.
 * Implements the same semantics as dfa_numpy.kernel.
 * The transition table is stored in row-major order: trans[state * NA + symbol].
 * Input arrays are ``int64`` (numpy int64) and are accessed via ``restrict`` pointers.
 * The kernel updates ``counts`` in-place, incrementing the visit count for each visited state.
 */

#include <stdint.h>
#include <stdio.h>

void dfa_fp64(const int64_t *restrict trans,
         const int64_t *restrict symbols,
         int64_t *restrict counts,
         int64_t NS,
         int64_t NA,
         int64_t N) {
    /* ``NS`` (number of states) is not needed for the core algorithm;
       it is present to match the driver’s expected signature. */
    (void)NS; // suppress unused parameter warning
        
    // Hint alignment for better SIMD / vector load performance.
    // trans alignment hint removed
    // symbols alignment hint removed
    // counts alignment hint removed
printf("NS=%lld NA=%lld N=%lld\n", (long long)NS, (long long)NA, (long long)N);

    int64_t state = 0;
    for (int64_t i = 0; i < N; ++i) {
        // Load symbol (already in range [0, NA-1] by initialization).
        int64_t sym = symbols[i];
        // Transition to the next state.
        state = trans[state * NA + sym];
        // Increment visit count for the new state.
        counts[state] += 1;
    }
}

