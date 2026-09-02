#include <stdint.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb, const double *restrict c,
                      int64_t LEN_2D,
                      uint8_t *restrict workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;
    const int64_t CHUNK = 128;

    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < LEN_2D; i0 += CHUNK) {
        int64_t i1 = i0 + CHUNK;
        if (i1 > LEN_2D) i1 = LEN_2D;

        for (int64_t i = i0; i < i1; ++i) {
            a[i] = a[i] + b[i] * c[i];
        }

        for (int64_t j = 1; j < LEN_2D; ++j) {
            double *restrict aa_prev = aa + (j - 1) * LEN_2D;
            double *restrict aa_cur  = aa + j * LEN_2D;
            const double *restrict bb_cur = bb + j * LEN_2D;
            #pragma omp simd
            for (int64_t i = i0; i < i1; ++i) {
                aa_cur[i] = aa_prev[i] + bb_cur[i] * a[i];
            }
        }
    }
}
