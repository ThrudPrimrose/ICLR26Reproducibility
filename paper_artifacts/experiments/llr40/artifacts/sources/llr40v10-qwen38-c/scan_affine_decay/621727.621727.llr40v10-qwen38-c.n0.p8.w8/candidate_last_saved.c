#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/*
 * y[i] = c[i]*y[i-1] + x[i], i=1..n-1; y[0] is the seed (set by caller).
 * ABI (C linkage): void scan_affine_decay_fp64(double *c, double *x, double *y,
 *            int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes);
 *
 * Tiled two-level blocked scan with tile-local prefix storage:
 *   Pass 1 (parallel over blocks, serial within block): compute the tile-local
 *           prefix LA[i]=prod c, LB[i]=forced part, AND each block's combined
 *           transform (GAc,GBc). LA/LB live in a tile-sized L3-resident buffer.
 *   Pass 2 (single thread over the tile's blocks): block input values Vin[].
 *   Pass 3 (fully parallel, vectorizable): y[i] = LA[i]*Vin + LB[i].
 *
 * Tiles are processed in order (each tile's input depends on the previous
 * tile's output). DRAM traffic = read c,x once + write y once = 24n bytes;
 * the only serial chain is pass 1 (one scan over n elements).
 */
void scan_affine_decay_fp64(double *c, double *x, double *y,
                            int64_t LEN_1D, uint8_t *workspace,
                            int64_t workspace_bytes)
{
    (void)workspace; (void)workspace_bytes;
    int64_t n = LEN_1D;
    if (n <= 1) return;
    int64_t M = n - 1;
    const int64_t B = 4096;          /* block size */
    const int64_t Tblocks = 128;     /* blocks per tile  -> T = 512K elems */
    const int64_t T = Tblocks * B;   /* max elements per tile */
    int64_t K = (M + B - 1) / B;
    int64_t nTiles = (K + Tblocks - 1) / Tblocks;

    double *GAc = (double *)malloc((size_t)K * sizeof(double));
    double *GBc = (double *)malloc((size_t)K * sizeof(double));
    double *Vin = (double *)malloc((size_t)K * sizeof(double));
    double *LA  = (double *)malloc((size_t)T * sizeof(double));
    double *LB  = (double *)malloc((size_t)T * sizeof(double));

    const double * __restrict cc  = c;
    const double * __restrict xx  = x;
    double *       __restrict lLA = LA;
    double *       __restrict lLB = LB;
    double *       __restrict lyy = y;
    double vin_shared = lyy[0];

    #pragma omp parallel
    {
        for (int64_t tile = 0; tile < nTiles; tile++) {
            int64_t b0 = tile * Tblocks;
            int64_t b1 = b0 + Tblocks;
            if (b1 > K) b1 = K;
            int64_t tstart = 1 + b0 * B;          /* first element (i) of tile */
            int64_t tend = (b1 - 1) * B + B - 1;
            if (tend >= n) tend = n - 1;

            /* Pass 1: tile-local prefix LA,LB + block transforms */
            #pragma omp for schedule(static)
            for (int64_t k = b0; k < b1; k++) {
                int64_t s = 1 + k * B;
                int64_t e = s + B - 1;
                if (e >= n) e = n - 1;
                double a = 1.0, b = 0.0;
                for (int64_t i = s; i <= e; i++) {
                    a = cc[i] * a;
                    b = cc[i] * b + xx[i];
                    lLA[i - tstart] = a;
                    lLB[i - tstart] = b;
                }
                GAc[k] = a;
                GBc[k] = b;
            }

            /* Pass 2: block input values within the tile (serial, one thread) */
            #pragma omp single
            {
                double vin = vin_shared;
                Vin[b0] = vin;
                for (int64_t k = b0 + 1; k < b1; k++) {
                    vin = GAc[k - 1] * vin + GBc[k - 1];
                    Vin[k] = vin;
                }
                vin_shared = GAc[b1 - 1] * vin + GBc[b1 - 1];
            }

            /* Pass 3: fully parallel, vectorizable */
            #pragma omp for schedule(static)
            for (int64_t k = b0; k < b1; k++) {
                double vin = Vin[k];
                int64_t s = 1 + k * B;
                int64_t e = s + B - 1;
                if (e >= n) e = n - 1;
                for (int64_t i = s; i <= e; i++) {
                    lyy[i] = lLA[i - tstart] * vin + lLB[i - tstart];
                }
            }
        }
    }

    free(GAc); free(GBc); free(Vin); free(LA); free(LB);
}
