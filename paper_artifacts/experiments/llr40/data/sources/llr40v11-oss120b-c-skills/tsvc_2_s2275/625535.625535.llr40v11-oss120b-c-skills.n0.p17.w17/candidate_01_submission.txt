/* Optimized version of TSVC tsvc_2 kernel "s2275" (fp64).
 * This implementation swaps loop order to achieve unit‑stride memory accesses,
 * separates the independent vector update of a, and adds OpenMP parallelism
 * with SIMD directives. The algorithmic behavior matches the reference.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa,
                       const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc,
                       const double *restrict d, const int64_t LEN_2D) {
    /* Update the 2‑D array aa in row‑major order (j outer, i inner).
     * Each element is independent, so we parallelise over rows (j).
     */
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int64_t j = 0; j < LEN_2D; ++j) {
            int64_t base = j * LEN_2D;
            #pragma omp simd
            for (int64_t i = 0; i < LEN_2D; ++i) {
                int64_t idx = base + i;
                aa[idx] = aa[idx] + bb[idx] * cc[idx];
            }
        }

        /* Independent vector operation on a, b, c, d. */
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            a[i] = b[i] + c[i] * d[i];
        }
    }
}
