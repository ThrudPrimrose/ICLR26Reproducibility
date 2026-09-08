#include <stdint.h>
#include <omp.h>

void tsvc_2_s255_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    const double inv3 = 0.333;
    int host = omp_get_initial_device();
    omp_set_default_device(host);
    #pragma omp target teams distribute parallel for simd map(to: b[0:LEN_1D]) map(from: a[0:LEN_1D])
    for (int64_t i = 0; i < LEN_1D; ++i) {
        int64_t im1 = i - 1;
        if (im1 < 0) im1 += LEN_1D;
        int64_t im2 = i - 2;
        if (im2 < 0) im2 += LEN_1D;
        a[i] = (b[i] + b[im1] + b[im2]) * inv3;
    }
}
