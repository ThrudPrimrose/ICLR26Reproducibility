/* Optimized version of TSVC tsvc_2 kernel s1232.
 * Original reference implementation (handported) performs:
 *   for (j = 0; j < LEN_2D; ++j)
 *     for (i = j*VLEN; i < LEN_2D; ++i)
 *       aa[i*LEN_2D + j] = bb[i*LEN_2D + j] + cc[i*LEN_2D + j];
 * The iteration domain satisfies i >= j*VLEN, which can be rewritten as
 *   for (i = 0; i < LEN_2D; ++i)
 *     for (j = 0; j <= i / VLEN; ++j)
 * This swap yields unit‑stride accesses in the inner loop, enabling
 * vectorisation and better cache utilisation. We also parallelise the outer
 * loop with OpenMP.
 */

#include <stdint.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                        const int64_t LEN_2D, const int64_t VLEN) {
    /* Guard against division by zero – VLEN is expected to be positive.
       If VLEN <= 0 we fall back to the original order (no work).
    */
    if (VLEN <= 0) return;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        int64_t j_max = i / VLEN; // floor division
        // The upper bound is always <= LEN_2D-1, no need to clamp.
        #pragma omp simd
        for (int64_t j = 0; j <= j_max; ++j) {
            int64_t idx = i * LEN_2D + j;
            aa[idx] = bb[idx] + cc[idx];
        }
    }
}
