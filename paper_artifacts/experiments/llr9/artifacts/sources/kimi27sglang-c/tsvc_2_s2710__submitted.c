#include <stddef.h>
#include <stdint.h>

void tsvc_2_s2710_fp64(double* restrict a, double* restrict b, double* restrict c,
                       double* restrict d, double* restrict e, double* restrict x,
                       int64_t LEN_1D, uint8_t* restrict workspace, int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;

    const int len_gt_10 = LEN_1D > 10;
    const double x0 = x[0];
    const int x_gt_0 = x0 > 0.0;

    #pragma omp simd
    for (int64_t i = 0; i < LEN_1D; i++) {
        const double ai = a[i];
        const double bi = b[i];
        const double ci = c[i];
        const double di = d[i];
        const double ei = e[i];

        const int cond = ai > bi;

        const double a_new = ai + bi * di;
        const double c_if = len_gt_10 ? (ci + di * di) : (di * ei + 1.0);

        const double b_new = ai + ei * ei;
        const double c_else = x_gt_0 ? (ai + di * di) : (ci + ei * ei);

        a[i] = cond ? a_new : ai;
        b[i] = cond ? bi : b_new;
        c[i] = cond ? c_if : c_else;
    }
}
