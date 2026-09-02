/* TSVC tsvc_2 s2710 -- branchless, vectorized, OpenMP-threaded.
 *
 * Reference semantics (per element i, all from original values):
 *   gt   = a[i] > b[i]
 *   a[i] = gt ? a[i] + b[i]*d[i] : a[i]
 *   b[i] = gt ? b[i]              : a[i] + e[i]*e[i]
 *   c[i] = gt ? (LEN_1D>10 ? c[i] + d[i]*d[i] : d[i]*e[i] + 1.0)
 *            : (x[0]>0.0  ? a[i] + d[i]*d[i]  : c[i] + e[i]*e[i])
 * No dependence across i; LEN_1D>10 and x[0]>0.0 are loop-invariant
 * (unswitched outside). Data-dependent gt becomes an exact select.
 */
#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double *restrict a,
                       double *restrict b,
                       double *restrict c,
                       const double *restrict d,
                       const double *restrict e,
                       const double *restrict x,
                       const int64_t LEN_1D,
                       uint8_t *const workspace,
                       const int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    const double x0 = (LEN_1D > 0) ? x[0] : 0.0;
    const int xpos = x0 > 0.0;
    const int len10 = LEN_1D > 10;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; i++) {
        const double ai = a[i];
        const double bi = b[i];
        const double ci = c[i];
        const double di = d[i];
        const double ei = e[i];
        const int gt = ai > bi;

        a[i] = gt ? ai + bi * di : ai;
        b[i] = gt ? bi : ai + ei * ei;
        if (len10)
            c[i] = gt ? ci + di * di : (xpos ? ai + di * di : ci + ei * ei);
        else
            c[i] = gt ? di * ei + 1.0 : (xpos ? ai + di * di : ci + ei * ei);
    }
}
