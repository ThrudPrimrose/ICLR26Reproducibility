#include <stdint.h>
#include <omp.h>

void tsvc_2_s2710_fp64(double * restrict a, double * restrict b, double * restrict c,
                       const double * restrict d, const double * restrict e, const double * restrict x,
                       int64_t LEN_1D, uint8_t * restrict workspace, int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;

    const int cond_len = (LEN_1D > 10);
    const int cond_x   = (x[0] > 0.0);

    if (cond_len) {
        if (cond_x) {
            #pragma omp parallel for simd schedule(static)
            for (int64_t i = 0; i < LEN_1D; i++) {
                const double ai = a[i];
                const double bi = b[i];
                const double ci = c[i];
                const double di = d[i];
                const double ei = e[i];
                const double mask = (ai > bi) ? 1.0 : 0.0;
                const double nmask = 1.0 - mask;
                a[i] = mask * (ai + bi * di) + nmask * ai;
                b[i] = mask * bi + nmask * (ai + ei * ei);
                c[i] = mask * (ci + di * di) + nmask * (ai + di * di);
            }
        } else {
            #pragma omp parallel for simd schedule(static)
            for (int64_t i = 0; i < LEN_1D; i++) {
                const double ai = a[i];
                const double bi = b[i];
                const double ci = c[i];
                const double di = d[i];
                const double ei = e[i];
                const double mask = (ai > bi) ? 1.0 : 0.0;
                const double nmask = 1.0 - mask;
                a[i] = mask * (ai + bi * di) + nmask * ai;
                b[i] = mask * bi + nmask * (ai + ei * ei);
                c[i] = mask * (ci + di * di) + nmask * (ci + ei * ei);
            }
        }
    } else {
        if (cond_x) {
            #pragma omp parallel for simd schedule(static)
            for (int64_t i = 0; i < LEN_1D; i++) {
                const double ai = a[i];
                const double bi = b[i];
                const double di = d[i];
                const double ei = e[i];
                const double mask = (ai > bi) ? 1.0 : 0.0;
                const double nmask = 1.0 - mask;
                a[i] = mask * (ai + bi * di) + nmask * ai;
                b[i] = mask * bi + nmask * (ai + ei * ei);
                c[i] = mask * (di * ei + 1.0) + nmask * (ai + di * di);
            }
        } else {
            #pragma omp parallel for simd schedule(static)
            for (int64_t i = 0; i < LEN_1D; i++) {
                const double ai = a[i];
                const double bi = b[i];
                const double ci = c[i];
                const double di = d[i];
                const double ei = e[i];
                const double mask = (ai > bi) ? 1.0 : 0.0;
                const double nmask = 1.0 - mask;
                a[i] = mask * (ai + bi * di) + nmask * ai;
                b[i] = mask * bi + nmask * (ai + ei * ei);
                c[i] = mask * (di * ei + 1.0) + nmask * (ci + ei * ei);
            }
        }
    }
}
