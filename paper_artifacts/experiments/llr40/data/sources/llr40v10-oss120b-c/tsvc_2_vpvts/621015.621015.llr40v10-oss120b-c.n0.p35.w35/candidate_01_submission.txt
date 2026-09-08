#include <stdint.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b,
                       const int64_t LEN_1D, const int64_t S) {
    // Cast scalar to double once for efficiency
    const double s = (double)S;
    // Parallelize across threads and vectorize the loop
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] += b[i] * s;
    }
}
