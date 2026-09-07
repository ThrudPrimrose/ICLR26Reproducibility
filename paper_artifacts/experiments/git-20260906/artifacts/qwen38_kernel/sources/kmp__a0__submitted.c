#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/* KMP substring search: count (overlapping) occurrences of `pattern` (len M)
 * in `text` (len N).  ABI: kmp_fp64(matches, pattern, text, M, N, ws, wsize).
 *
 * Fast path: bit-parallel Shift-And for M <= 64.  D is an M-bit mask; after
 * consuming text[i], bit j of D is set iff pattern[0..j] == text[i-j..i].  A
 * full match (bit M-1 set) ending at i is counted.  Per symbol:
 *     D = ((D<<1)|1) & M(t)
 * where M(t) has bit j set iff pattern[j]==t (0 for a value not in the
 * pattern).  Parallelized over disjoint chunks; each chunk pre-scans the M-1
 * symbols before it so cross-boundary matches (counted by END position) are
 * found; chunks partition [0,N) so every match is counted exactly once.
 */
void kmp_fp64(int64_t *matches, const int64_t *pattern, const int64_t *text,
              int64_t M, int64_t N, uint8_t *workspace, int64_t wsize)
{
    if (M <= 0 || N <= 0 || M > N) { *matches = 0; return; }

    int binary = 1;
    uint64_t mask0 = 0, mask1 = 0;
    if (M <= 64) {
        for (int64_t i = 0; i < M; i++) {
            if (pattern[i] == 0) mask0 |= (1ULL << (unsigned)i);
            else if (pattern[i] == 1) mask1 |= (1ULL << (unsigned)i);
            else { binary = 0; break; }
        }
    } else {
        binary = 0;
    }

    if (binary) {
        uint64_t topbit = 1ULL << (unsigned)(M - 1);
        int P = omp_get_max_threads();
        if (P < 1) P = 1;
        int64_t base = N / P, rem = N % P;
        int64_t total = 0;
        #pragma omp parallel for reduction(+:total) schedule(static)
        for (int64_t p = 0; p < P; p++) {
            int64_t lo = p * base + (p < rem ? p : rem);
            int64_t hi = lo + base + (p < rem ? 1 : 0);
            int64_t start = lo - (M - 1); if (start < 0) start = 0;
            uint64_t D = 0;
            for (int64_t i = start; i < lo; i++) {
                int64_t t = text[i];
                uint64_t m = ((((uint64_t)(-(int64_t)(t == 0))) & mask0) |
                              (((uint64_t)(-(int64_t)(t == 1))) & mask1));
                D = ((D << 1) | 1ULL) & m;
            }
            int64_t c = 0;
            for (int64_t i = lo; i < hi; i++) {
                int64_t t = text[i];
                uint64_t m = ((((uint64_t)(-(int64_t)(t == 0))) & mask0) |
                              (((uint64_t)(-(int64_t)(t == 1))) & mask1));
                D = ((D << 1) | 1ULL) & m;
                c += (int64_t)((D & topbit) != 0);
            }
            total += c;
        }
        *matches = total;
        return;
    }

    /* Fallback: classical KMP (M > 64 or non-binary pattern).  Single-threaded. */
    uint32_t *fail = (uint32_t *)malloc((size_t)M * sizeof(uint32_t));
    if (!fail) { *matches = 0; return; }
    fail[0] = 0;
    uint32_t k = 0;
    for (int64_t i = 1; i < M; i++) {
        while (k > 0 && pattern[k] != pattern[i]) k = fail[k - 1];
        if (pattern[k] == pattern[i]) k++;
        fail[i] = k;
    }
    int64_t count = 0;
    uint32_t q = 0;
    for (int64_t i = 0; i < N; i++) {
        while (q > 0 && pattern[q] != text[i]) q = fail[q - 1];
        if (pattern[q] == text[i]) q++;
        if (q == (uint32_t)M) { count++; q = fail[M - 1]; }
    }
    *matches = count;
    free(fail);
}
