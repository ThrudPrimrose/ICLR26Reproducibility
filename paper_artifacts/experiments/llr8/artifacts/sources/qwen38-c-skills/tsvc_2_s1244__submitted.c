#include <stdint.h>
#include <omp.h>

/*
 * TSVC s1244 (numpy reference):
 *   for i in 0..LEN_1D-2:
 *       a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]
 *       d[i] = a[i](new) + a[i+1](original)
 *
 * The read a[i+1] in d[i] is an anti-dependence: iteration i+1 overwrites it,
 * so the read means the ORIGINAL value. We never write a while computing d:
 *   pass 1 (parallel): d[i] = f(i) + a[i+1]   (a still original)
 *   pass 2 (parallel): a[i] = f(i)
 * f(i) = b[i] + c[i]*c[i] + b[i]*b[i] + c[i] is recomputed in both passes --
 * cheap FLOPs, saves a full N-double buffer and one N-read.
 * Total traffic: 7N doubles (56 B/elem) vs 5N for the serial baseline.
 */
void tsvc_2_s1244_fp64(double *a, double *b, double *c, double *d,
                       int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    const int64_t n = LEN_1D - 1; /* trip count: i = 0 .. LEN_1D-2 */
    if (n <= 0)
        return;

    if (n >= (1 << 15)) {
        /* pass 1 reads the original a; pass 2 writes a later (barrier between). */
        const double *restrict pa = a;
        const double *restrict pb = b;
        const double *restrict pc = c;
        double *restrict pd = d;
        double *restrict pa_w = a;

        #pragma omp parallel
        {
            #pragma omp for simd schedule(static)
            for (int64_t i = 0; i < n; i++) {
                const double x = pb[i] + pc[i] * pc[i] + pb[i] * pb[i] + pc[i];
                pd[i] = x + pa[i + 1];
            }
            #pragma omp for simd schedule(static)
            for (int64_t i = 0; i < n; i++)
                pa_w[i] = pb[i] + pc[i] * pc[i] + pb[i] * pb[i] + pc[i];
        }
    } else {
        const double *restrict pa = a;
        const double *restrict pb = b;
        const double *restrict pc = c;
        double *restrict pd = d;
        double *restrict pa_w = a;

        for (int64_t i = 0; i < n; i++) {
            const double x = pb[i] + pc[i] * pc[i] + pb[i] * pb[i] + pc[i];
            pd[i] = x + pa[i + 1];
        }
        for (int64_t i = 0; i < n; i++)
            pa_w[i] = pb[i] + pc[i] * pc[i] + pb[i] * pb[i] + pc[i];
    }
}
