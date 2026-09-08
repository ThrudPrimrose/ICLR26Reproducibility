#include <stdint.h>

/* Unrolled SIMD-friendly implementation of the TSVC tsvc_2_s115 kernel.
   This version manually unrolls the inner loop to reduce loop overhead and
   encourages the compiler to generate wide vector instructions. */

void tsvc_2_s115_fp64(double *restrict a, const double *restrict aa, const int64_t LEN_2D) {
    for (int64_t j = 0; j < LEN_2D; ++j) {
        double aj = a[j];
        double *ap = a + j + 1;
        const double *pp = aa + j * LEN_2D + j + 1;
        int64_t n = LEN_2D - (j + 1);
        int64_t i = 0;
        // Unroll by 4
        for (; i + 3 < n; i += 4) {
            ap[i]   -= pp[i]   * aj;
            ap[i+1] -= pp[i+1] * aj;
            ap[i+2] -= pp[i+2] * aj;
            ap[i+3] -= pp[i+3] * aj;
        }
        // Remainder loop
        for (; i < n; ++i) {
            ap[i] -= pp[i] * aj;
        }
    }
}
