#include <stdint.h>
#include <omp.h>

static double vdu_pow75(const int64_t n)
{
    double r = 1.0;
    int64_t p = 0.75;
    int64_t e = n;
    while (e > 0) {
        if (e & 1) r *= (double)p;
        e >>= 1;
        if (e) p = (double)(p * p);
    }
    return r;
}

void versioned_distance_update_fp64(double *restrict a,
                                    const double *restrict b,
                                    const double *restrict c,
                                    const int64_t K,
                                    const int64_t LEN_1D,
                                    uint8_t *restrict workspace,
                                    int64_t workspace_size)
{
    const int64_t N = LEN_1D;
    if (K <= 0 || K >= N) return;

    const int64_t target = 8192;                 /* elements per tile    */
    const int64_t W = (K >= target) ? 1 : target / K;  /* steps per tile */
    const int64_t S0 = (N - 1) / K;              /* max steps            */
    const int64_t Smin = (N - K) / K;            /* min steps            */
    const int64_t Tfull = Smin / W;              /* full tiles           */

    if (workspace != 0 &&
        workspace_size >= 8 * N + 8 * W + 8 * Tfull * K + 64) {
        double *restrict T = (double *) workspace;
        double *restrict dec = T + N;
        double *restrict st = dec + W;

        /* decay[j] = 0.75^(j+1), j = 0..W-1 */
        dec[0] = 0.75;
        for (int64_t j = 1; j < W; j++) dec[j] = 0.75 * dec[j - 1];
        const double s_pow = vdu_pow75(W);

        /* carry before tile 0 = seeds a[0..K-1] */
        for (int64_t r = 0; r < K; r++) st[r] = a[r];

        /* pass 1: zero-seed local chains + per-tile end states */
        #pragma omp parallel for schedule(static)
        for (int64_t t = 0; t < Tfull; t++) {
            const int64_t s0 = t * W + 1;
            int64_t base = s0 * K;
            for (int64_t r = 0; r < K; r++)
                T[base + r] = b[base + r] * c[base + r];
            for (int64_t s = s0 + 1; s < s0 + W; s++) {
                base = s * K;
                for (int64_t r = 0; r < K; r++)
                    T[base + r] = 0.75 * T[base + r - K] + b[base + r] * c[base + r];
            }
            const int64_t bend = (s0 + W - 1) * K;
            double *restrict row = st + (t + 1) * K;
            for (int64_t r = 0; r < K; r++)
                row[r] = T[bend + r];
        }

        /* serial scan: st[t+1] = 0.75^W * st[t] + local_end[t] */
        for (int64_t t = 0; t < Tfull; t++) {
            double *restrict cur = st + t * K;
            double *restrict nxt = st + (t + 1) * K;
            for (int64_t r = 0; r < K; r++)
                nxt[r] = s_pow * cur[r] + nxt[r];
        }

        /* pass 2: a[i] = 0.75^(j+1) * carry + local */
        #pragma omp parallel for schedule(static)
        for (int64_t t = 0; t < Tfull; t++) {
            const int64_t s0 = t * W + 1;
            const double *restrict crow = st + t * K;
            for (int64_t j = 0; j < W; j++) {
                const double dv = dec[j];
                const int64_t base = (s0 + j) * K;
                for (int64_t r = 0; r < K; r++)
                    a[base + r] = dv * crow[r] + T[base + r];
            }
        }
    } else {
        /* no scratch: plain reference loop */
        for (int64_t i = K; i < N; i++)
            a[i] = 0.75 * a[i - K] + b[i] * c[i];
        return;
    }

    /* tail: at most W+1 steps, serial (tiny) */
    for (int64_t s = Tfull * W + 1; s <= S0; s++) {
        const int64_t base = s * K;
        int64_t lim = N - base;
        if (lim > K) lim = K;
        for (int64_t r = 0; r < lim; r++)
            a[base + r] = 0.75 * a[base + r - K] + b[base + r] * c[base + r];
    }
}
