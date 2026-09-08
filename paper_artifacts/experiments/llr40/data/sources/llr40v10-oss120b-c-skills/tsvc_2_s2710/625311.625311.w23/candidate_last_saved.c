/* Optimized version of tsvc_2_s2710_fp64 kernel.
 * This version adds OpenMP parallelism, hoists invariants, and reduces redundant work.
 */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
    const bool len_gt_10 = (LEN_1D > 10);
    const bool x0_pos = (x[0] > 0.0);

    #pragma omp parallel for simd schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ai = a[i];
        double bi = b[i];
        double di = d[i];
        double ei = e[i];
        double di2 = di * di;
        double ei2 = ei * ei;
        if (ai > bi) {
            a[i] = ai + bi * di;
            if (len_gt_10) {
                c[i] += di2;
            } else {
                c[i] = di * ei + 1.0;
            }
        } else {
            b[i] = ai + ei2;
            if (x0_pos) {
                c[i] = ai + di2;
            } else {
                c[i] += ei2;
            }
        }
    }
}
