#include <stdint.h>
#include <stdbool.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
    const bool len_gt_10 = LEN_1D > 10;
    const bool x0_pos = x[0] > 0.0;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (a[i] > b[i]) {
            a[i] += b[i] * d[i];
            if (len_gt_10) {
                c[i] += d[i] * d[i];
            } else {
                c[i] = d[i] * e[i] + 1.0;
            }
        } else {
            b[i] = a[i] + e[i] * e[i];
            if (x0_pos) {
                c[i] = a[i] + d[i] * d[i];
            } else {
                c[i] += e[i] * e[i];
            }
        }
    }
}
