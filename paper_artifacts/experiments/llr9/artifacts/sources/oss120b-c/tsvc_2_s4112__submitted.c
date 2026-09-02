#include <omp.h>
#include <stddef.h>
#include <stdint.h>

// TSVC tsvc_2 kernel s4112: a[i] = a[i] + 2.0 * b[ip[i]]
// Expected entry point: void tsvc_2_s4112_fp64(double *a, double *b, int32_t *ip,
//                                           int64_t LEN_1D, uint8_t *workspace,
//                                           int64_t workspace_bytes);
// The workspace arguments are ignored.

#ifdef __cplusplus
extern "C" {
#endif

void tsvc_2_s4112_fp64(double * restrict a, double * restrict b, int32_t * restrict ip,
                       int64_t LEN_1D, uint8_t * /*workspace*/, int64_t /*workspace_bytes*/) {
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = a[i] + b[(int64_t)ip[i]] * 2.0;
    }
}

#ifdef __cplusplus
}
#endif