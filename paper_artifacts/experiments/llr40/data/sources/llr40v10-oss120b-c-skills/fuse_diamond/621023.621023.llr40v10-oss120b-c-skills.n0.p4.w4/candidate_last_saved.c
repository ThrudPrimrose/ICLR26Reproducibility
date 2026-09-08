#include <stdint.h>
#include <omp.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double x = a[i];
        double x2 = x * x;
        out[i] = x2 * x2 - 1.0;
    }
}
