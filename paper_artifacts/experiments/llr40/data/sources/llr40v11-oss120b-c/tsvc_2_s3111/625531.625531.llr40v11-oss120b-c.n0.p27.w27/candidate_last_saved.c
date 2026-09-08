/* Optimized version of tsvc_2_s3111_fp64 kernel.
 * Computes sum of positive elements of input array a, stores result in b[0].
 * Uses OpenMP parallel for simd with reduction to exploit multi-core and vectorization.
 */

#include <stdint.h>

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    double sum = 0.0;
    #pragma omp parallel for simd reduction(+:sum) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double val = a[i];
        sum += (val > 0.0) ? val : 0.0; // branchless conditional addition
    }
    b[0] = sum;
}
