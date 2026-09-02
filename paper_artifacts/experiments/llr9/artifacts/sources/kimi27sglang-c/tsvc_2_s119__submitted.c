#include <stdint.h>

void tsvc_2_s119_fp64(double *restrict aa, double *restrict bb, int64_t LEN_2D,
                      uint8_t *restrict workspace, int64_t workspace_bytes) {
    #pragma omp parallel
    {
        for (int64_t i = 1; i < LEN_2D; ++i) {
            double *restrict aa_i = aa + i * LEN_2D;
            const double *restrict aa_im1 = aa + (i - 1) * LEN_2D;
            const double *restrict bb_i = bb + i * LEN_2D;
            #pragma omp for simd
            for (int64_t j = 1; j < LEN_2D; ++j) {
                aa_i[j] = aa_im1[j - 1] + bb_i[j];
            }
        }
    }
}
