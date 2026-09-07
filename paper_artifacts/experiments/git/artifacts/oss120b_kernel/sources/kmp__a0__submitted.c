/* KMP substring search kernel
 * Implements the reference algorithm from kmp_numpy.py.
 * Constructs the prefix failure table and scans the text.
 * Arguments:
 *   text    - pointer to int64_t array of length N
 *   pattern - pointer to int64_t array of length M
 *   matches - pointer to int64_t array of length 1 (output count)
 *   N       - length of text
 *   M       - length of pattern
 */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

void kmp_fp64(const int64_t *restrict text, const int64_t *restrict pattern,
         int64_t *restrict matches, int64_t N, int64_t M, uint8_t *restrict workspace, int64_t workspace_bytes) {
    // Debug: function entry

        // Debug: log arguments to /tmp/kmp_debug.txt
    FILE *debug_fp = fopen("/shared/agent-6/kmp_debug.txt", "a");
    if (debug_fp) {
        fprintf(debug_fp, "kmp_fp64 entry: N=%ld M=%ld text=%p pattern=%p matches=%p workspace=%p wsbytes=%ld\n",
                (long)N, (long)M, (void*)text, (void*)pattern, (void*)matches, (void*)workspace, (long)workspace_bytes);
        fflush(debug_fp);
    }
    // Edge case: empty pattern
    if (M <= 0) {
        matches[0] = 0;
        if (debug_fp) fclose(debug_fp);
        return;
    }

    // Early exit: pattern longer than text cannot match
    if (M > N) {
        matches[0] = 0;
        if (debug_fp) fclose(debug_fp);
        return;
    }
    // Allocate failure table using dynamic memory (ignore workspace)
    int64_t *fail = (int64_t *)malloc((size_t)M * sizeof(int64_t));
    if (!fail) {
        matches[0] = 0;
        if (debug_fp) fclose(debug_fp);
        return;
    }
    // Build prefix failure table
    fail[0] = 0;
    int64_t k = 0;
    for (int64_t i = 1; i < M; ++i) {
        while (k > 0 && pattern[k] != pattern[i]) {
            k = fail[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            ++k;
        }
        fail[i] = k;
    }
    // Scan the text
    int64_t count = 0;
    int64_t q = 0;
    for (int64_t i = 0; i < N; ++i) {
        while (q > 0 && pattern[q] != text[i]) {
            q = fail[q - 1];
        }
        if (pattern[q] == text[i]) {
            ++q;
        }
        if (q == M) {
            ++count;
            q = fail[q - 1];
        }
    }
    matches[0] = count;
    if (debug_fp) {
        fprintf(debug_fp, "Result count: %ld\n", (long)count);
        fflush(debug_fp);
    }
    free(fail);
    if (debug_fp) fclose(debug_fp);
}
