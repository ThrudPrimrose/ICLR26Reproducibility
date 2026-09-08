#include <stdint.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
    const double s = (double)S;
    if (LEN_1D > 1500000) {
        #pragma omp parallel for schedule(guided, 32768)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] += b[i] * s;
        }
    } else {
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] += b[i] * s;
        }
    }
}
