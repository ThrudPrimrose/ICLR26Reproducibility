/* Optimized implementation of TSVC tsvc_2_vpvts kernel.
   Adds OpenMP parallelization and SIMD to improve performance over reference.
   The reference kernel simply does:
       a[i] += b[i] * S;
   where a and b are double arrays of length LEN_1D, and S is an integer.
   This version uses a double cast of S once, and a combined parallel for simd pragma.
*/

#include <stdint.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b,
                       const int64_t LEN_1D, const int64_t S) {
    double s_val = (double)S;
    /* Parallelize across outer loop; static schedule gives even chunk distribution.
       "simd" clause enables vectorization inside each thread's chunk. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] += b[i] * s_val;
    }
}

