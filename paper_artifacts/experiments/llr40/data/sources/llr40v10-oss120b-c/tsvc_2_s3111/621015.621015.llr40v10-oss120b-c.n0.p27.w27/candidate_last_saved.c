/* Optimized version of tsvc_2_s3111 kernel.
 * Computes sum of positive elements in array a and stores result in b[0].
 * Uses OpenMP parallel reduction for speed.
 */
#include <stdint.h>

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        if (ai > 0.0) {
            sum += ai;
        }
    }
    b[0] = sum;
}
