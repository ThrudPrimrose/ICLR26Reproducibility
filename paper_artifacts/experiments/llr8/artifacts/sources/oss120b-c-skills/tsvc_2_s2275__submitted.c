/* Optimized version of tsvc_2_s2275_fp64 kernel.
 * Apply loop interchange for better memory locality,
 * parallelize outer loop, and vectorize inner loop.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {
    // Process the 2D arrays: aa[idx] += bb[idx] * cc[idx]
    // Interchange loops to make inner loop stride-1 (contiguous) access.
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < LEN_2D; ++j) {
        int64_t base = j * LEN_2D;
        #pragma omp simd
        for (int64_t i = 0; i < LEN_2D; ++i) {
            int64_t idx = base + i;
            aa[idx] = aa[idx] + bb[idx] * cc[idx];
        }
    }
    // Process the 1D vectors.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] = b[i] + c[i] * d[i];
    }
}
