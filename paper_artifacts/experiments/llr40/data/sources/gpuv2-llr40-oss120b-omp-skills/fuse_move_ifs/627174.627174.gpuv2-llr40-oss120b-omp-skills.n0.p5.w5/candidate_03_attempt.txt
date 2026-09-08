/* Optimized version of fuse_move_ifs kernel with OpenMP offloading.
   Implements the same functionality as the reference kernel, but
   parallelizes work on the GPU using target teams distribute parallel for.
   The kernels are vectorized via the simd clause.
*/

#include <stdint.h>
#include <omp.h>

void fuse_move_ifs_fp64(double *restrict a, double *restrict b,
                         const double *restrict cond, const double *restrict src,
                         const int64_t K, const int64_t LEN_2D) {
    // Offload data to the device once.
    #pragma omp target data map(to: src[0:LEN_2D*LEN_2D], cond[0:LEN_2D]) \
                                map(tofrom: a[0:LEN_2D*LEN_2D], b[0:LEN_2D*LEN_2D])
    {
        // Compute array a where cond[i] > 0.
        #pragma omp target teams distribute parallel for simd collapse(2)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            for (int64_t j = 0; j < LEN_2D; ++j) {
                if (cond[i] > 0.0) {
                    a[i * LEN_2D + j] = src[i * LEN_2D + j] * 2.0;
                }
            }
        }
        // Compute array b if K > 0.
        if (K > 0) {
            #pragma omp target teams distribute parallel for simd collapse(2)
            for (int64_t i = 0; i < LEN_2D; ++i) {
                for (int64_t j = 0; j < LEN_2D; ++j) {
                    b[i * LEN_2D + j] = src[i * LEN_2D + j] + 1.0;
                }
            }
        }
    }
}
