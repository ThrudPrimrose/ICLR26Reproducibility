/* Optimized version of TSVC 2 s275 kernel for OpenMP offload.
 * The original reference implementation performs a column‑wise recurrence:
 *   for i in [0, LEN_2D):
 *     if (aa[i] > 0.0)
 *       for j in [1, LEN_2D):
 *         aa[j*LEN_2D + i] = aa[(j-1)*LEN_2D + i] + bb[j*LEN_2D + i] * cc[j*LEN_2D + i];
 *
 * The outer loop over columns (i) has no inter‑column dependencies, so we offload it to the
 * GPU using an OpenMP target region. The inner loop remains sequential because of the
 * recurrence across rows (j). The whole matrices are mapped to the device; the transfer cost
 * is part of the timed region.
 */

#include <stdint.h>

void tsvc_2_s275_fp64(double *restrict aa,
                       const double *restrict bb,
                       const double *restrict cc,
                       const int64_t LEN_2D) {
    /* Offload the column loop to the device.
     * Map the full arrays; the mapping cost is included in the benchmark timing.
     */
    #pragma omp target map(to: bb[0:LEN_2D*LEN_2D], cc[0:LEN_2D*LEN_2D]) map(tofrom: aa[0:LEN_2D*LEN_2D])
    {
        #pragma omp teams distribute parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            double prev = aa[i]; // first element of column i (aa[0,i])
            if (prev > 0.0) {
                double *aa_col = aa + i;
                const double *bb_col = bb + i;
                const double *cc_col = cc + i;
                for (int64_t j = 1; j < LEN_2D; ++j) {
                    double val = prev + bb_col[j * LEN_2D] * cc_col[j * LEN_2D];
                    aa_col[j * LEN_2D] = val;
                    prev = val;
                }
            }
        }
    }
}
