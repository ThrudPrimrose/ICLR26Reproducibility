/* Optimized version of TSVC tsvc_2_s2275 kernel.
 * Original reference performs a 2D nested loop updating aa and a separate vector update for a.
 * This version flattens the 2D loop for better memory locality, uses OpenMP parallelism,
 * and enables vectorization on both loops.
 */
#include <stdint.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa,
                       const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc,
                       const double *restrict d,
                       const int64_t LEN_2D) {
    // Compute a[i] = b[i] + c[i] * d[i]
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] = b[i] + c[i] * d[i];
    }
    // Flattened update of aa: aa[idx] = aa[idx] + bb[idx] * cc[idx]
    const int64_t total = LEN_2D * LEN_2D;
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total; ++idx) {
        aa[idx] += bb[idx] * cc[idx];
    }
}
