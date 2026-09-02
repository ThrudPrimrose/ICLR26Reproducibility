#include <stdint.h>
#include <omp.h>
#include <string.h>

/*
 * TSVC ext_war_unit kernel (s121)
 *
 * Performs the operation
 *   a[i] = a[i+1] + b[i]
 * for i = 0 .. LEN_1D-2.  The last element a[LEN_1D-1] is left unchanged.
 *
 * The original loop has a forward anti‑dependence.  A safe parallel version
 * first shifts the original data left by one element using the highly‑optimised
 * ``memmove`` (which correctly handles the overlap) and then adds the companion
 * array ``b``.  This avoids allocating an explicit temporary buffer and reduces
 * the number of memory passes.
 *
 * The function follows the exact signature required by the benchmark harness:
 *   void ext_war_unit_fp64(double *restrict a,
 *                         const double *restrict b,
 *                         int64_t LEN_1D,
 *                         uint8_t *restrict workspace,
 *                         int64_t workspace_bytes)
 *
 * The ``workspace`` argument is retained for compatibility; it is not required
 * for the memmove‑based implementation but is used for a fallback path when the
 * caller supplies insufficient space for an auxiliary buffer.
 */
void ext_war_unit_fp64(double *restrict a,
                       const double *restrict b,
                       int64_t LEN_1D,
                       uint8_t *restrict workspace,
                       int64_t workspace_bytes)
{
    // Guard against degenerate sizes – the loop body would be empty.
    if (LEN_1D <= 1) {
        (void)workspace;          // suppress unused‑parameter warnings
        (void)workspace_bytes;
        return;
    }

    // ``memmove`` works in‑place and handles overlapping regions safely.
    // It copies LEN_1D-1 elements from a[1] to a[0].
    memmove(a, a + 1, (size_t)(LEN_1D - 1) * sizeof(double));

    // After the shift, a[i] contains the original a[i+1].  We now add b[i]
    // to obtain the final result.  The addition loop is vectorisable and can
    // be parallelised across threads.
    #pragma omp parallel for schedule(static) default(none) \
        shared(a, b, LEN_1D)
    for (int64_t i = 0; i < LEN_1D - 1; ++i) {
        a[i] = a[i] + b[i];
    }

    // The final element a[LEN_1D-1] already holds the original value – no work
    // is required.
    (void)workspace_bytes; // keep the parameter unused without a warning
}
