/* Optimized version of tsvc_2_s2275 kernel.
 * Original reference performs a 2-D element-wise update of aa using bb and cc,
 * and computes a[i] = b[i] + c[i] * d[i].
 * This implementation:
 *   - Interchanges loops for unit‑stride memory access.
 *   - Uses a single OpenMP parallel region with two work‑sharing loops to avoid
 *     repeated parallel region entry overhead.
 *   - Applies SIMD vectorisation to the inner loops.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa,
                       const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc,
                       const double *restrict d, const int64_t LEN_2D) {
    #pragma omp parallel
    {
        /* Matrix update: aa = aa + bb * cc (element‑wise)
         * Outer loop over rows (j) gives unit‑stride inner loop over columns (i).
         */
        #pragma omp for schedule(static) nowait
        for (int64_t j = 0; j < LEN_2D; ++j) {
            double *restrict aa_row = aa + j * LEN_2D;
            const double *restrict bb_row = bb + j * LEN_2D;
            const double *restrict cc_row = cc + j * LEN_2D;
            #pragma omp simd
            for (int64_t i = 0; i < LEN_2D; ++i) {
                aa_row[i] += bb_row[i] * cc_row[i];
            }
        }
        /* Compute a = b + c * d */
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            a[i] = b[i] + c[i] * d[i];
        }
    }
}
