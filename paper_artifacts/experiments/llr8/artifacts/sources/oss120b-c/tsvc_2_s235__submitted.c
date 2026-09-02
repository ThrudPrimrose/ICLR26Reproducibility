/* Parallel optimized implementation of TSVC tsvc_2 kernel s235 (fp64). */
#include <stdint.h>
void tsvc_2_s235_fp64(double *restrict a, double *restrict aa, const double *restrict b, const double *restrict bb, const double *restrict c, const int64_t LEN_2D) {
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_2D; ++i) {
            a[i] += b[i] * c[i];
        }
        for (int64_t j = 1; j < LEN_2D; ++j) {
            const double *bb_row = bb + j * LEN_2D;
            const double *aa_prev = aa + (j - 1) * LEN_2D;
            double *aa_cur = aa + j * LEN_2D;
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < LEN_2D; ++i) {
                aa_cur[i] = aa_prev[i] + bb_row[i] * a[i];
            }
        }
    }
}
