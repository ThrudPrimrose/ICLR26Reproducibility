#include <stdint.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        double *aa_col = aa + i;
        const double *bb_col = bb + i;
        const double *cc_col = cc + i;
        double prev = aa_col[0]; // aa[0][i]
        if (prev > 0.0) {
            for (int64_t j = 1; j < LEN_2D; ++j) {
                double prod = bb_col[j * LEN_2D] * cc_col[j * LEN_2D];
                double cur = prev + prod;
                aa_col[j * LEN_2D] = cur;
                prev = cur;
            }
        }
    }
}
