/* Optimized version of tsvc_2_s233 using OpenMP parallelism.
 * Original reference computes two independent scans across a 2D array.
 * The scans are independent per column (aa) and per row (bb), allowing
 * parallelism across the outer dimension of each scan.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    // Parallel scan of aa across j for each column i.
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int64_t i = 8; i < LEN_2D; ++i) {
            for (int64_t j = 8; j < LEN_2D; ++j) {
                aa[j * LEN_2D + i] = aa[(j - 1) * LEN_2D + i] + cc[j * LEN_2D + i];
            }
        }
        // Parallel scan of bb across i for each row j.
        #pragma omp for schedule(static)
        for (int64_t j = 8; j < LEN_2D; ++j) {
            for (int64_t i = 8; i < LEN_2D; ++i) {
                bb[j * LEN_2D + i] = bb[j * LEN_2D + (i - 1)] + cc[j * LEN_2D + i];
            }
        }
    }
}

