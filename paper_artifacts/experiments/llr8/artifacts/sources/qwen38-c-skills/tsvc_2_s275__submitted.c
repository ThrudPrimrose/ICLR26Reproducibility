#include <stdint.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *aa, double *bb, double *cc, int64_t LEN_2D,
                      uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_2D < 2)
        return;

    double *restrict a = aa;
    const double *restrict b = bb;
    const double *restrict c = cc;
    const double *restrict row0 = aa;

    #pragma omp parallel
    {
        const int64_t nt = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();
        const int64_t base = LEN_2D / nt;
        const int64_t rem = LEN_2D % nt;
        const int64_t i0 = tid * base + (tid < rem ? tid : rem);
        const int64_t i1 = i0 + base + (tid < rem ? 1 : 0);
        for (int64_t j = 1; j < LEN_2D; j++) {
            const double *restrict prev = a + (j - 1) * LEN_2D;
            double *restrict cur = a + j * LEN_2D;
            const double *restrict brow = b + j * LEN_2D;
            const double *restrict crow = c + j * LEN_2D;
            for (int64_t i = i0; i < i1; i++) {
                const double old = cur[i];
                const double v = prev[i] + brow[i] * crow[i];
                cur[i] = (row0[i] > 0.0) ? v : old;
            }
        }
    }
}
