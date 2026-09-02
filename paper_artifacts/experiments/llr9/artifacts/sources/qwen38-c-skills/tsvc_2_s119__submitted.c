#include <stdint.h>
#include <stddef.h>
#include <omp.h>

/*
 * TSVC s119:  for i in 1..N-1: for j in 1..N-1:
 *                aa[i,j] = aa[i-1,j-1] + bb[i,j]
 *
 * The only dependence is aa[i-1,j-1] (upper-left, dependence vector (1,1)),
 * so every axis carries it and the row-major nest is a serial wavefront.
 *
 * We tile the (i,j) plane into B x B blocks and sweep the block-diagonals
 * (s = ti + tj) from 0 outward.  A tile (ti,tj) reads aa[i-1,j-1] which is
 * either in the same tile (previous row/col, already computed in row-major
 * order within the tile) or in a strictly earlier block-diagonal (s-1 or
 * s-2).  Hence all tiles on one block-diagonal are mutually independent and
 * are computed in parallel.  Inside a tile the inner j loop is unit-stride
 * (row i write, row i-1 read offset by -1, row i of bb) and vectorizes.
 */
void tsvc_2_s119_fp64(double *aa, double *bb, int64_t LEN_2D,
                      uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    const int64_t N = LEN_2D;
    if (N < 2)
        return;

    const int64_t nthr = omp_get_max_threads();
    int64_t B = N / (2 * nthr);
    if (B < 32)
        B = 32;
    if (B > 256)
        B = 256;
    if (B > N)
        B = N;

    const int64_t R = (N + B - 1) / B; /* tile rows */
    const int64_t C = (N + B - 1) / B; /* tile cols */
    const int64_t Smax = R + C - 1;

    #pragma omp parallel
    {
        for (int64_t s = 0; s < Smax; s++) {
            int64_t ti_lo = s - (C - 1);
            if (ti_lo < 0)
                ti_lo = 0;
            int64_t ti_hi = s < (R - 1) ? s : (R - 1);
            #pragma omp for schedule(static)
            for (int64_t ti = ti_lo; ti <= ti_hi; ti++) {
                const int64_t tj = s - ti;
                int64_t i0 = ti * B;
                if (i0 < 1)
                    i0 = 1;
                int64_t i1 = (ti + 1) * B;
                if (i1 > N)
                    i1 = N;
                int64_t j0 = tj * B;
                if (j0 < 1)
                    j0 = 1;
                int64_t j1 = (tj + 1) * B;
                if (j1 > N)
                    j1 = N;
                for (int64_t i = i0; i < i1; i++) {
                    double *__restrict aa_i = aa + i * N;
                    const double *__restrict aa_im1 = aa + (i - 1) * N;
                    const double *__restrict bb_i = bb + i * N;
                    for (int64_t j = j0; j < j1; j++)
                        aa_i[j] = aa_im1[j - 1] + bb_i[j];
                }
            }
        }
    }
}
