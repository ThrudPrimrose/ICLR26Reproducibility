#include <stdint.h>
#include <omp.h>

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b, const int32_t *restrict ip, const int64_t LEN_1D) {
    if (LEN_1D >= (1LL << 20)) {
        #pragma omp parallel for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] += b[ip[i]] * 2.0;
        }
    } else {
        #pragma omp simd
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] += b[ip[i]] * 2.0;
        }
    }
}
