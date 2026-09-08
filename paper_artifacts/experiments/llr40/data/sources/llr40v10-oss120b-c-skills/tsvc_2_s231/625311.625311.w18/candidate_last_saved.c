/* Optimized parallel version of tsvc_2_s231_fp64
 * - Parallelizes outer loop over columns (i) with OpenMP.
 * - Rewrites recurrence to use a scalar accumulator, reducing memory traffic.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    // Parallelize across the i dimension: each column is independent.
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        // Load initial value from aa[0][i]
        double sum = aa[i]; // j = 0 element
        // Compute prefix sum over bb for this column and store to aa.
        for (int64_t j = 1; j < LEN_2D; ++j) {
            sum += bb[j * LEN_2D + i];
            aa[j * LEN_2D + i] = sum;
        }
    }
}

