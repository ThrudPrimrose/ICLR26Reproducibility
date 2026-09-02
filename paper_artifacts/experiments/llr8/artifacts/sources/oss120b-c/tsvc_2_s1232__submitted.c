#include <omp.h>
#include <stdint.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, int64_t LEN_2D, int64_t VLEN, uint8_t *workspace, int64_t workspace_bytes) {
    (void)workspace; (void)workspace_bytes;
    // Guard against zero VLEN to avoid division by zero
    if (VLEN <= 0) {
        // If VLEN is non-positive, the original loops would start at i = j*VLEN;
        // with non-positive VLEN, j*VLEN is <=0 for all j, so inner loop spans the whole row.
        // Implement as full copy of bb+cc into aa.
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            double *aa_row = aa + i * LEN_2D;
            const double *bb_row = bb + i * LEN_2D;
            const double *cc_row = cc + i * LEN_2D;
            #pragma omp simd
            for (int64_t j = 0; j < LEN_2D; ++j) {
                aa_row[j] = bb_row[j] + cc_row[j];
            }
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        int64_t max_j = i / VLEN;
        if (max_j >= LEN_2D) max_j = LEN_2D - 1;
        double *aa_row = aa + i * LEN_2D;
        const double *bb_row = bb + i * LEN_2D;
        const double *cc_row = cc + i * LEN_2D;
        #pragma omp simd
        for (int64_t j = 0; j <= max_j; ++j) {
            aa_row[j] = bb_row[j] + cc_row[j];
        }
    }
}
