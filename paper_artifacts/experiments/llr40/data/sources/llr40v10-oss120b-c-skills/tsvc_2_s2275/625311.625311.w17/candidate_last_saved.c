#include <stdint.h>
#include <omp.h>

void tsvc_2_s2275_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb,
                       const double *restrict c, const double *restrict cc, const double *restrict d,
                       const int64_t LEN_2D) {
    // Compute the vector a[i] = b[i] + c[i] * d[i]
    // Use parallel for with SIMD for vectorization.
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        a[i] = b[i] + c[i] * d[i];
    }

    // Update aa with aa[idx] = aa[idx] + bb[idx] * cc[idx]
    // Reorder loops to make inner loop unit-stride for better memory access.
    #pragma omp parallel for simd schedule(static)
    for (int64_t idx = 0; idx < LEN_2D * LEN_2D; ++idx) {
        aa[idx] = aa[idx] + bb[idx] * cc[idx];
    }
}
