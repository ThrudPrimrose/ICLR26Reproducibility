#include <stddef.h>
#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

/*
 * TSVC_2 kernel s323 implementation in C (double precision).
 * Reference (Python/NumPy) signature:
 *   def s323(a, b, c, d, e, LEN_1D):
 *       a[i] = b[i - 1] + c[i] * d[i]
 *       b[i] = a[i] + c[i] * e[i]
 * The arrays are double‑precision and stored in contiguous (C‑style) layout.
 *
 * This implementation transforms the recurrence to reduce the amount of work and to
 * expose vectorisation opportunities.  It uses the supplied workspace as a temporary
 * buffer for the intermediate term `c[i] * (d[i] + e[i])`.
 *
 * The function follows the standard ABI expected by the benchmark harness:
 *   void tsvc_2_s323_fp64(double *restrict a,
 *                        double *restrict b,
 *                        const double *restrict c,
 *                        const double *restrict d,
 *                        const double *restrict e,
 *                        const int64_t LEN_1D,
 *                        uint8_t *restrict workspace,
 *                        const int64_t workspace_bytes);
 */

void tsvc_2_s323_fp64(double *restrict a,
                       double *restrict b,
                       const double *restrict c,
                       const double *restrict d,
                       const double *restrict e,
                       const int64_t LEN_1D,
                       uint8_t *restrict workspace,
                       const int64_t workspace_bytes)
{
    // No work needed for LEN_1D <= 1 (loop starts at i = 1)
    if (LEN_1D <= 1) {
        return;
    }

    // Optimized sequential loop using pointer arithmetic to reduce indexing overhead.
    // Preserves the exact floating‑point order required for correctness.
    double prev = b[0];
    double *ap = a + 1;
    double *bp = b + 1;
    const double *cp = c + 1;
    const double *dp = d + 1;
    const double *ep = e + 1;
    int64_t n = LEN_1D - 1;
    for (int64_t i = 0; i < n; ++i) {
        // Prefetch next cache lines to hide memory latency.
        if (i + 16 < n) {
            __builtin_prefetch(cp + 16);
            __builtin_prefetch(dp + 16);
            __builtin_prefetch(ep + 16);
        }
        double ci = *cp++;
        double di = *dp++;
        double ei = *ep++;
        double ai = prev + ci * di;
        *ap++ = ai;
        double bi = ai + ci * ei;
        *bp++ = bi;
        prev = bi;
    }
}
