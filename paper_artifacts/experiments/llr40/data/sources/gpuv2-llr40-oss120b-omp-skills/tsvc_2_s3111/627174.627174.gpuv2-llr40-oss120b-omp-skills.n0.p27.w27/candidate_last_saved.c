/* Optimized version of tsvc_2_s3111 kernel.
 * Computes sum of positive values in array a and stores result in b[0].
 * Uses OpenMP parallel for with SIMD and reduction for multi-core vectorized performance.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    double sum = 0.0;
    // Parallelize across threads and vectorize inner loop.
    #pragma omp target teams distribute parallel for simd reduction(+:sum) map(to: a[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        // Add only positive values; use conditional to aid vectorization.
        sum += (ai > 0.0) ? ai : 0.0;
    }
    b[0] = sum;
}
