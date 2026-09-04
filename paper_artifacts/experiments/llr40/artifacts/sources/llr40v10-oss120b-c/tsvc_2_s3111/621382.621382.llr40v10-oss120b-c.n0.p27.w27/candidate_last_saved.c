/* Optimized version of tsvc_2_s3111 kernel.
   Computes sum of positive elements in array a, writes to b[0].
   Uses OpenMP parallel for reduction and SIMD vectorization. */

#include <stdint.h>

/* The function name and signature must match the reference. */
void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    double sum = 0.0;
    /* Parallelize across threads and let the compiler vectorize.
       The reduction clause safely accumulates the partial sums.
       schedule(static) provides even distribution.
    */
    #pragma omp parallel for simd reduction(+:sum) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        if (ai > 0.0) {
            sum += ai;
        }
    }
    b[0] = sum;
}

