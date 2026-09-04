#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
    const int len_gt_10 = (LEN_1D > 10);
    const double x0 = x[0];
    const int x0pos = (x0 > 0.0);
    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        const double ai = a[i];
        const double bi = b[i];
        const double di = d[i];
        const double ei = e[i];
        if (ai > bi) {
            a[i] = ai + bi * di;
            if (len_gt_10) {
                c[i] += di * di;
            } else {
                c[i] = di * ei + 1.0;
            }
        } else {
            b[i] = ai + ei * ei;
            if (x0pos) {
                c[i] = ai + di * di;
            } else {
                c[i] += ei * ei;
            }
        }
    }
}
