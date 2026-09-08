/* Branchless version with outer conditionals and OpenMP parallel for */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
    const int len_gt_10 = (LEN_1D > 10);
    const int x_pos = (x[0] > 0.0);

    if (len_gt_10) {
        if (x_pos) {
            #pragma omp parallel for schedule(static) if(LEN_1D > 1024)
            for (int64_t i = 0; i < LEN_1D; ++i) {
                double ai = a[i];
                double bi = b[i];
                double di = d[i];
                double ei = e[i];
                double ci = c[i];
                int gt = (ai > bi);
                double a_new = ai + bi * di;
                double b_new = ai + ei * ei;
                double c_new_if = ci + di * di;      // case a[i] > b[i]
                double c_new_else = ai + di * di;    // case a[i] <= b[i]
                a[i] = gt ? a_new : ai;
                b[i] = gt ? bi : b_new;
                c[i] = gt ? c_new_if : c_new_else;
            }
        } else {
            #pragma omp parallel for schedule(static) if(LEN_1D > 1024)
            for (int64_t i = 0; i < LEN_1D; ++i) {
                double ai = a[i];
                double bi = b[i];
                double di = d[i];
                double ei = e[i];
                double ci = c[i];
                int gt = (ai > bi);
                double a_new = ai + bi * di;
                double b_new = ai + ei * ei;
                double c_new_if = ci + di * di;      // a[i] > b[i]
                double c_new_else = ci + ei * ei;    // a[i] <= b[i]
                a[i] = gt ? a_new : ai;
                b[i] = gt ? bi : b_new;
                c[i] = gt ? c_new_if : c_new_else;
            }
        }
    } else {
        // LEN_1D <= 10: use straightforward logic (small loop, negligible cost)
        #pragma omp parallel for schedule(static) if(LEN_1D > 1024)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double ai = a[i];
            double bi = b[i];
            double di = d[i];
            double ei = e[i];
            double ci = c[i];
            if (ai > bi) {
                a[i] = ai + bi * di;
                c[i] = di * ei + 1.0;
            } else {
                b[i] = ai + ei * ei;
                c[i] = x_pos ? (ai + di * di) : (ci + ei * ei);
            }
        }
    }
}

