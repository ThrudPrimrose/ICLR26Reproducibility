#include <stdint.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    // Parallelize and vectorize the element-wise multiplication of three arrays.
    // Use OpenMP parallel for with SIMD and aligned data hints for best performance.
    #pragma omp parallel for simd aligned(a,b,c:64)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }
}
