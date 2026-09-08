/* Optimized version of tsvc_2_s4112 microkernel.
   Performs a[i] += 2.0 * b[ip[i]] for i = 0..LEN_1D-1.
   The operation is embarrassingly parallel across i, so we use OpenMP.
   The kernel uses restrict pointers for aliasing assumptions.
   Added OpenMP parallel for to exploit multi-core CPU.
   Vectorization with gather is left to the compiler.
*/

#include <stdint.h>

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    #pragma omp target teams distribute parallel for simd map(to: b[0:LEN_1D], ip[0:LEN_1D]) map(tofrom: a[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] += b[ip[i]] * 2.0;
    }
}
