/* Optimized version of tsvc_2_s252_fp64
 * Original reference computes running sum with dependence, rewritten to eliminate
 * the loop-carried dependence and enable parallelization.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    // First element without dependence
    a[0] = b[0] * c[0];
    // Parallel loop for remaining elements
    #pragma omp parallel for simd
    for (int64_t i = 1; i < LEN_1D; ++i) {
        double cur = b[i] * c[i];
        // Use previous element's product, independent of loop-carried state
        a[i] = cur + b[i-1] * c[i-1];
    }
}
