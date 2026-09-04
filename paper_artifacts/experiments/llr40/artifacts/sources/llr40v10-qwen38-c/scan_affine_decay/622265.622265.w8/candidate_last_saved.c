/* scan_affine_decay: y[i] = c[i]*y[i-1] + x[i]  (affine scan, variable coefficient)
 *
 * Two-level blocked scan.  Block bi covers steps k = i0+1 .. i0+len (i0 = bi*S,
 * len <= S), reading c[x] at the SAME indices and starting from y[i0].
 * Waves of W blocks keep their c,x lines L3-resident between phase 1 and 3.
 *  phase 1: per block, serial compute of the affine map (a,b) of the block
 *           (y_out = a*y_in + b); store the intra-wave EXCLUSIVE prefix per
 *           block and the wave total.
 *  phase 2: serial scan over the per-wave totals (a few thousand entries),
 *           converting them to exclusive wave prefixes.
 *  phase 3: per block, recompute the intra-block prefix and apply the
 *           (wave o block) prefix to write y; c,x re-read from L3.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define S 64L        /* steps per block   */
#define W 256L       /* blocks per wave   */

static inline void compose(double *a, double *b, double a2, double b2) {
    /* (*a,*b) <- (a2,b2) o (*a,*b): t -> a2*((*a)*t+(*b)) + b2 */
    double a0 = *a;
    *a = a2 * a0;
    *b = a2 * (*b) + b2;
}

void scan_affine_decay_fp64(const double *restrict c, const double *restrict x,
                            double *restrict y, const int64_t LEN_1D,
                            uint8_t *restrict workspace, const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    if (LEN_1D <= 1) return;

    const int64_t nsteps = LEN_1D - 1;
    const int64_t nblk = (nsteps + S - 1) / S;
    const int64_t nwav = (nblk + W - 1) / W;

    double *buf = (double *)malloc((size_t)(nblk * 2 + nwav * 2) * sizeof(double));
    double *blk_a = buf;
    double *blk_b = buf + nblk;
    double *wav_a = buf + 2 * nblk;
    double *wav_b = buf + 2 * nblk + nwav;

    /* phase 1: block totals + intra-wave exclusive prefixes */
    #pragma omp parallel for schedule(static)
    for (int64_t w = 0; w < nwav; w++) {
        const int64_t b0 = w * W;
        const int64_t b1 = b0 + W < nblk ? b0 + W : nblk;
        double wa = 1.0, wb = 0.0;
        for (int64_t bi = b0; bi < b1; bi++) {
            const int64_t i0 = bi * S + 1;             /* first step index */
            const int64_t len = i0 + S <= LEN_1D ? S : LEN_1D - i0;
            const double *cp = c + i0;
            const double *xp = x + i0;
            double a = 1.0, b = 0.0;
            for (int64_t j = 0; j < len; j++) {
                a *= cp[j];
                b = cp[j] * b + xp[j];
            }
            blk_a[bi] = wa;
            blk_b[bi] = wb;
            compose(&wa, &wb, a, b);
        }
        wav_a[w] = wa;
        wav_b[w] = wb;
    }

    /* phase 2: serial scan over wave totals -> exclusive wave prefixes */
    {
        double ra = 1.0, rb = 0.0;
        for (int64_t w = 0; w < nwav; w++) {
            const double A = wav_a[w], B = wav_b[w];
            wav_a[w] = ra;
            wav_b[w] = rb;
            ra = A * ra;
            rb = A * rb + B;
        }
    }

    /* phase 3: write y from local prefixes + (wave o block) prefix */
    {
        const double y0 = y[0];
        #pragma omp parallel for schedule(static)
        for (int64_t w = 0; w < nwav; w++) {
            const int64_t b0 = w * W;
            const int64_t b1 = b0 + W < nblk ? b0 + W : nblk;
            const double ra = wav_a[w], rb = wav_b[w];
            for (int64_t bi = b0; bi < b1; bi++) {
                const int64_t i0 = bi * S + 1;
                const int64_t len = i0 + S <= LEN_1D ? S : LEN_1D - i0;
                const double *cp = c + i0;
                const double *xp = x + i0;
                double *yp = y + i0;
                const double A = blk_a[bi];
                const double B = blk_b[bi];
                const double V = (ra * A) * y0 + (ra * B + rb);
                double aa = 1.0, bb = 0.0;
                for (int64_t j = 0; j < len; j++) {
                    aa *= cp[j];
                    bb = cp[j] * bb + xp[j];
                    yp[j] = aa * V + bb;
                }
            }
        }
    }
    free(buf);
}
