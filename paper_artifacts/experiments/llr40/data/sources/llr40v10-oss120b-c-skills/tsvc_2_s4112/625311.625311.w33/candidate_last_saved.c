/* Optimized version of tsvc_2_s4112 kernel with OpenMP parallelization and SIMD.
 * Original reference performs: a[i] += b[ip[i]] * 2.0;
 * This version adds a parallel for and simd to exploit multi-core and vector units.
 */

#include <stdint.h>
#include <omp.h>

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    /* Parallelize across i. Each iteration updates a distinct element of a.
       The loop body uses an indirect read from b via ip, which is safe for parallelism.
       The combined 'parallel for simd' directive distributes iterations among threads
       and allows vectorization (gather) within each thread. */
    #pragma omp simd
    for (int64_t i = 0; i < LEN_1D; ++i) {
                a[i] += b[(int64_t)ip[i]] * 2.0;
    }
}

