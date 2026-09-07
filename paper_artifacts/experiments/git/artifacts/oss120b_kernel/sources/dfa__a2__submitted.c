#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void dfa_fp64(const int64_t *restrict trans, const int64_t *restrict symbols, int64_t *restrict counts, int64_t N, int64_t NA, int64_t NS, uint8_t *restrict workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    printf("N=%lld NA=%lld NS=%lld\n", (long long)N, (long long)NA, (long long)NS);
    // Ensure counts are zeroed at start to avoid mismatches if caller does not zero-initialize.
    for (int64_t i = 0; i < NS; ++i) {
        counts[i] = 0;
    }
    int64_t state = 0;
    for (int64_t i = 0; i < N; ++i) {
        int64_t sym = symbols[i];
        state = trans[state * NA + sym];
        counts[state] += 1;
    }

}

#ifdef __cplusplus
}
#endif
