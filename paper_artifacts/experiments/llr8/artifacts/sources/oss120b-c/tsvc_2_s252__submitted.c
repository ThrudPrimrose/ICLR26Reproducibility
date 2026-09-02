/*
 * TSVC tsvc_2_s252 kernel – C implementation
 *
 * The reference Python implementation (NumPy) computes:
 *   t = 0.0
 *   for i in range(LEN_1D):
 *       s = b[i] * c[i]
 *       a[i] = s + t
 *       t = s
 *
 * This can be rewritten mathematically as:
 *   a[0] = b[0] * c[0]
 *   a[i] = b[i] * c[i] + b[i-1] * c[i-1]   for i >= 1
 *
 * The rewritten form removes the loop‑carried scalar dependence, enabling
 * vectorisation and parallelisation.  The implementation below follows this
 * transformation and uses OpenMP to parallelise the independent iterations.
 */

#include <stddef.h>
#include <stdint.h>

/*
 * Kernel entry point.
 *
 * Parameters:
 *   a        – output array of length LEN_1D
 *   b, c    – input arrays of length LEN_1D
 *   LEN_1D  – number of elements in each array
 *
 * The driver supplies contiguous, 64‑byte aligned buffers.  All pointers are
 * assumed to be non‑NULL and to point to at least LEN_1D elements.
 */
void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D <= 0)
        return;
    a[0] = b[0] * c[0];
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 1; i < LEN_1D; ++i) {
        a[i] = b[i] * c[i] + b[i - 1] * c[i - 1];
    }
}

