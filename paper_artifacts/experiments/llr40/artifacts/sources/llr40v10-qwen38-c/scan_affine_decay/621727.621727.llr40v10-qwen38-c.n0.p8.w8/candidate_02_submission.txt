#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/*
 * y[i] = c[i]*y[i-1] + x[i], i=1..n-1; y[0] is the seed (set by caller).
 * ABI (C linkage): void scan_affine_decay_fp64(double *c, double *x, double *y,
 *            int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes);
 *
 * Tiled two-level blocked scan. Blocks (size B) are grouped into tiles
 * (Tblocks each) so a tile's c+x working set stays resident in L3, making the
 * second (pass 3) read of c,x a cache hit. Tiles are processed in order
 * (each tile's input value depends on the previous tile's output); within a
 * tile the blocks are parallel.
 *
 *   Pass 1 (parallel over tile's blocks): block combined transform (GAc,GBc).
 *   Pass 2 (single thread over tile's blocks): block input values Vin[].
 *   Pass 3 (parallel over tile's blocks): re-run recurrence from Vin[], write y.
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
    int64_t K = (M + B - 1) / B;
    int64_t nTiles = (K + Tblocks - 1) / Tblocks;

    double *GAc = (double *)malloc((size_t)K * sizeof(double));
    double *GBc = (double *)malloc((size_t)K * sizeof(double));
    double *Vin = (double *)malloc((size_t)K * sizeof(double));

    const double * __restrict cc = c;
    const double * __restrict xx = x;
    double * __restrict yy = y;
    double vin_shared = yy[0];       /* value entering the current tile */

    #pragma omp parallel
    {
        for (int64_t tile = 0; tile < nTiles; tile++) {
            int64_t b0 = tile * Tblocks;
            int64_t b1 = b0 + Tblocks;
            if (b1 > K) b1 = K;

            /* Pass 1: block combined transforms  y[e]=GAc*y[s-1]+GBc */
            #pragma omp for schedule(static)
            for (int64_t k = b0; k < b1; k++) {
                int64_t s = 1 + k * B;
                int64_t e = s + B - 1;
                if (e >= n) e = n - 1;
                double a = 1.0, b = 0.0;
                for (int64_t i = s; i <= e; i++) {
                    b = cc[i] * b + xx[i];
                    a = cc[i] * a;
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

            /* Pass 3: re-run recurrence within each block, write y */
            #pragma omp for schedule(static)
            for (int64_t k = b0; k < b1; k++) {
                int64_t s = 1 + k * B;
                int64_t e = s + B - 1;
                if (e >= n) e = n - 1;
                double vv = Vin[k];
                for (int64_t i = s; i <= e; i++) {
                    vv = cc[i] * vv + xx[i];
                    yy[i] = vv;
                }
            }
        }
    }

    free(GAc); free(GBc); free(Vin);
}
