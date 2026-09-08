#include <stdint.h>
#include <omp.h>

#define MAXT 1024

/* Block-wise stream compaction.
 * Phase A: each block counts its survivors (pure reduction -> vectorizes).
 * Phase B: exclusive prefix of block counts, then each block scans its own
 *          predicate with an unrolled local counter and scatters the products.
 */

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 const double *restrict src,
                                 const double *restrict weight,
                                 const int64_t LEN_1D,
                                 uint8_t *restrict workspace,
                                 const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    const int64_t N = LEN_1D;
    if (N <= 0) {
        *out_count = 0;
        return;
    }

    int T = omp_get_max_threads();
    if (T > MAXT) T = MAXT;
    if (T < 1) T = 1;

    /* small problem: single thread, no fork cost */
    if (T == 1 || N < (1 << 18)) {
        int64_t n = 0;
        for (int64_t i = 0; i < N; ++i) {
            if (src[i] > 0.0) {
                packed[n] = src[i] * weight[i];
                n += 1;
            }
        }
        *out_count = n;
        return;
    }

    static int64_t cnt[MAXT + 1]; /* block counts -> exclusive prefix, then total in cnt[nb] */

    const int64_t B = (N + T - 1) / T;
    int nb = (int)((N + B - 1) / B);
    if (nb > T) nb = T;

#pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();

        /* ---- phase A: per-block survivor counts ---- */
        for (int b = tid; b < nb; b += T) {
            int64_t lo = (int64_t)b * B;
            int64_t hi = lo + B;
            if (hi > N) hi = N;
            int64_t c = 0;
            for (int64_t i = lo; i < hi; ++i)
                c += (src[i] > 0.0) ? 1 : 0;
            cnt[b] = c;
        }

        /* ---- prefix of block counts (thread 0; nb <= 1024) ---- */
#pragma omp barrier
        if (tid == 0) {
            int64_t s = 0;
            for (int b = 0; b < nb; ++b) {
                int64_t t = cnt[b];
                cnt[b] = s;
                s += t;
            }
            cnt[nb] = s;
        }
#pragma omp barrier

        /* ---- phase B: local scan + scatter ---- */
        for (int b = tid; b < nb; b += T) {
            int64_t lo = (int64_t)b * B;
            int64_t hi = lo + B;
            if (hi > N) hi = N;
            double *pk = packed + cnt[b];
            int64_t c = 0;
            int64_t i = lo;
            for (; i + 7 < hi; i += 8) {
                const double *sp = src + i;
                const double *wp = weight + i;
                double v0 = sp[0] * wp[0];
                double v1 = sp[1] * wp[1];
                double v2 = sp[2] * wp[2];
                double v3 = sp[3] * wp[3];
                double v4 = sp[4] * wp[4];
                double v5 = sp[5] * wp[5];
                double v6 = sp[6] * wp[6];
                double v7 = sp[7] * wp[7];
                int k0 = sp[0] > 0.0;
                int k1 = sp[1] > 0.0;
                int k2 = sp[2] > 0.0;
                int k3 = sp[3] > 0.0;
                int k4 = sp[4] > 0.0;
                int k5 = sp[5] > 0.0;
                int k6 = sp[6] > 0.0;
                int k7 = sp[7] > 0.0;
                int w = 0;
                if (k0) pk[c + w] = v0;
                w += k0;
                if (k1) pk[c + w] = v1;
                w += k1;
                if (k2) pk[c + w] = v2;
                w += k2;
                if (k3) pk[c + w] = v3;
                w += k3;
                if (k4) pk[c + w] = v4;
                w += k4;
                if (k5) pk[c + w] = v5;
                w += k5;
                if (k6) pk[c + w] = v6;
                w += k6;
                if (k7) pk[c + w] = v7;
                w += k7;
                c += w;
            }
            for (; i < hi; ++i) {
                if (src[i] > 0.0)
                    pk[c++] = src[i] * weight[i];
            }
        }
    }

    *out_count = cnt[nb];
}
