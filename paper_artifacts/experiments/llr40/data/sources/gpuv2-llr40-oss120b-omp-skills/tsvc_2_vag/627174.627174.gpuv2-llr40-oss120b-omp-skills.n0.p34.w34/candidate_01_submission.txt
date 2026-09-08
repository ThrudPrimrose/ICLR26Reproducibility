/* Optimized version of tsvc_2_vag_fp64 for OpenMP offload.
 * Performs a gather from b using index array ip and stores into a.
 * Uses a target region to offload work to the GPU and maps the arrays.
 */
#include <stdint.h>
#include <omp.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    // Offload the loop to the device. Map the input and output arrays.
    // b and ip are read-only, a is written.
    #pragma omp target teams distribute parallel for simd schedule(static) \
        map(to: b[0:LEN_1D]) map(to: ip[0:LEN_1D]) map(tofrom: a[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = b[(int64_t)ip[i]];
    }
}
