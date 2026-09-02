#include <stdint.h>
#include <stddef.h>

/*
 * TSVC 2 kernel s3112 (fp64).
 * Expected entry point signature:
 *   void tsvc_2_s3112_fp64(double *a, double *b,
 *                           int64_t LEN_1D,
 *                           uint8_t *workspace, int64_t workspace_bytes);
 * The workspace arguments are unused but required for ABI compatibility.
 *
 * This implementation follows the reference algorithm exactly (sequential
 * prefix sum) but uses manual loop unrolling to reduce loop overhead and aid the
 * compiler's optimizer.  The intent is to gain modest speedups while preserving
 * bit‑identical results.
 */

void tsvc_2_s3112_fp64(double *a, double *b,
                       int64_t LEN_1D,
                       uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D <= 0) {
        return;
    }
    double sum = 0.0;
    int64_t i = 0;
    int64_t n = LEN_1D;
    // Process 4 elements per iteration (unrolled).
    int64_t limit = n - (n % 4); // largest multiple of 4 <= n
    for (; i < limit; i += 4) {
        sum += a[i];
        b[i] = sum;
        sum += a[i + 1];
        b[i + 1] = sum;
        sum += a[i + 2];
        b[i + 2] = sum;
        sum += a[i + 3];
        b[i + 3] = sum;
    }
    // Tail loop for remaining elements.
    for (; i < n; ++i) {
        sum += a[i];
        b[i] = sum;
    }
}

