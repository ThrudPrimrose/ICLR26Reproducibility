/* Optimized version of TSVC tsvc_2 kernel s2275 (fp64).
 * Refactored for better memory locality and parallelism.
 */
#include <stdint.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa,
                       const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc,
                       const double *restrict d, const int64_t LEN_2D) {
    // Compute a[i] = b[i] + c[i] * d[i] in parallel.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] = b[i] + c[i] * d[i];
    }

    // Update aa[j,i] = aa[j,i] + bb[j,i] * cc[j,i] for all (j,i).
    // Use loop interchange for unit stride accesses and parallelism.
    #pragma omp parallel for schedule(static) collapse(2)
    for (int64_t j = 0; j < LEN_2D; ++j) {
        for (int64_t i = 0; i < LEN_2D; ++i) {
            int64_t idx = j * LEN_2D + i;
            aa[idx] = aa[idx] + bb[idx] * cc[idx];
        }
    }
}
