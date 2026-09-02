#include <stdint.h>
#include <omp.h>

/* ABI (binding order): a, aa, b, bb, c, cc, d, LEN_2D, workspace, workspace_size */
void tsvc_2_s2275_fp64(double *a, double *aa, double *b, double *bb,
                       double *c, double *cc, double *d, int64_t LEN_2D,
                       uint8_t *workspace, int64_t workspace_size)
{
    (void)workspace; (void)workspace_size;
    const int64_t N2 = LEN_2D * LEN_2D;

    #pragma omp parallel
    {
        double *restrict paa = aa;
        double *restrict pbb = bb;
        double *restrict pcc = cc;
        double *restrict pa = a;
        double *restrict pb = b;
        double *restrict pc = c;
        double *restrict pd = d;

        #pragma omp for schedule(static)
        for (int64_t k = 0; k < N2; k++)
            paa[k] = pbb[k] * pcc[k] + paa[k];

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < LEN_2D; i++)
            pa[i] = pb[i] + pc[i] * pd[i];
    }
}
