// hpcagent_bench-autogen -- generated from versioned_distance_update_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override.
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

/* a[i] = 0.75 * a[i-K] + b[i] * c[i]   (i = K .. LEN_1D-1)
 *
 * The dependence runs at runtime distance K: K independent chains (chain r:
 * i = r, r+K, r+2K, ...), each the scalar recurrence x_t = 0.75 x_{t-1} + y_t,
 * y_t = b[i]*c[i].  We decouple the chains into D=128-step blocks.  The decay
 * gives 0.75^128 ~ 1e-16, far below the fp64 tolerance, so the carry into a
 * block from the one before it is numerically invisible and every block is
 * independent -- fully parallel.
 *
 * Block m of chain r holds the chain steps t in [mD, (m+1)D).  Its boundary
 * value is B_m = 0.75^D B_{m-1} + S_m where
 *   S_m = sum_{t=(m-1)D+1}^{mD} 0.75^{mD-t} y_t
 * i.e. S_m is the y-recurrence over the PREVIOUS block run from 0.  Since
 * 0.75^D B_{m-1} ~ 1e-16, B_m ~= S_m.  Two phases per block:
 *   phase 1 (m >= 1): run the previous block's y-recurrence from 0  -> S_m
 *   phase 2 (all m): run this block's y-recurrence from S_m (or the seed
 *                    a[r] for m = 0), storing a.
 *
 * Blocks are linearized m-major (L = m*K + r) so that W consecutive blocks
 * are W consecutive chains at the same depth: at each step the threads touch
 * W *consecutive* doubles in b, c and a (compact, cache-friendly) while W
 * independent recurrences hide FMA/memory latency.  W is tuned on K.
 */

#define VDU_D 128   /* re-seed period in chain steps (0.75^128 ~ 1e-16) */

static void vdu_blocks(double *restrict a, const double *restrict b,
                       const double *restrict c, const int64_t K,
                       const int64_t LEN_1D, int W, int nt) {
    int64_t totalB = 0;
    for (int64_t r = 0; r < K; r++) {
        int64_t T_r = (LEN_1D - 1 - r) / K;
        totalB += (T_r + VDU_D - 1) / VDU_D;
    }
    if (totalB == 0) return;
    const int64_t base = totalB / nt;
    const int64_t rem  = totalB % nt;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int64_t L0 = (int64_t)tid * base + ((int64_t)tid < rem ? (int64_t)tid : rem);
        const int64_t L1 = L0 + base + (((int64_t)tid < rem) ? 1 : 0);

        for (int64_t L = L0; L < L1; L += W) {
            int64_t i2[8];
            int64_t len8[8];
            double x8[8];
            int p1[8];
            int64_t mx = 0;
            int any_p1 = 0;
            int nblk = 0;

            for (int w = 0; w < W && L + w < L1; w++) {
                const int64_t Lw = L + w;
                const int64_t m  = Lw / K;
                const int64_t r  = Lw % K;
                const int64_t T_r = (LEN_1D - 1 - r) / K;
                const int64_t t0 = m * VDU_D;
                if (t0 >= T_r) continue;
                int64_t len = T_r - t0;
                if (len > VDU_D) len = VDU_D;
                i2[nblk]   = r + (t0 + 1) * K;
                len8[nblk] = len;
                x8[nblk]   = (m == 0) ? a[r] : 0.0;
                p1[nblk]   = (m != 0);
                any_p1    |= (m != 0);
                if (len > mx) mx = len;
                nblk++;
            }

            /* phase 1 (m >= 1): S_m from the previous block's y (full D steps) */
            if (any_p1) {
                int64_t i1[8];
                for (int w = 0; w < nblk; w++) i1[w] = i2[w] - (int64_t)VDU_D * K;
                for (int64_t s = 0; s < VDU_D; s++) {
                    for (int w = 0; w < nblk; w++) {
                        if (!p1[w]) continue;
                        const int64_t i = i1[w];
                        x8[w] = 0.75 * x8[w] + b[i] * c[i];
                        i1[w] = i + K;
                    }
                }
            }
            /* phase 2: from S_m (or the seed), storing a */
            for (int64_t s = 0; s < mx; s++) {
                for (int w = 0; w < nblk; w++) {
                    if (s >= len8[w]) continue;
                    const int64_t i = i2[w];
                    double x = 0.75 * x8[w] + b[i] * c[i];
                    a[i] = x;
                    x8[w] = x;
                    i2[w] = i + K;
                }
            }
        }
    }
}

void versioned_distance_update_fp64(double *restrict a, const double *restrict b,
                                    const double *restrict c, const int64_t K,
                                    const int64_t LEN_1D) {
    if (K < 1 || LEN_1D <= K) return;
    const int nt = omp_get_max_threads();
    const int W = (K <= 2) ? 2 : 4;
    vdu_blocks(a, b, c, K, LEN_1D, W, nt);
}
