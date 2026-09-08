#include <stdint.h>
#include <omp.h>

void tsvc_2_vag_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    #pragma omp target data map(to: b[0:LEN_1D]) map(to: ip[0:LEN_1D]) map(from: a[0:LEN_1D])
    {
        #pragma omp target teams distribute parallel for simd
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] = b[ip[i]];
        }
    }
}
