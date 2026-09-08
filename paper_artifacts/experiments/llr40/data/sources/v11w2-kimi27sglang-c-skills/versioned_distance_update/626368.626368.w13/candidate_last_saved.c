#include <stdint.h>
#include <omp.h>

void versioned_distance_update_fp64(double *restrict a,
                                    double *restrict b,
                                    double *restrict c,
                                    int64_t K,
                                    int64_t LEN_1D,
                                    uint8_t *restrict workspace,
                                    int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    if (K <= 0 || K >= LEN_1D) {
        return;
    }

    if (K == 1) {
        for (int64_t i = 1; i < LEN_1D; i++) {
            a[i] = 0.75 * a[i - 1] + b[i] * c[i];
        }
        return;
    }

    /* Small problems: just vectorise across the K independent lanes in waves. */
    if (LEN_1D <= 8192) {
        for (int64_t base = K; base < LEN_1D; base += K) {
            int64_t end = base + K;
            if (end > LEN_1D) end = LEN_1D;
            for (int64_t i = base; i < end; i++) {
                a[i] = 0.75 * a[i - K] + b[i] * c[i];
            }
        }
        return;
    }

    /* Large problems: parallelise over contiguous residue ranges so that each
       thread streams through memory, vectorising across its residues within a wave. */
    int64_t nt = omp_get_max_threads();
    int64_t chunk = (K + nt - 1) / nt;
    if (chunk < 8) chunk = 8;
    /* round up to a multiple of 8 doubles (64 bytes) to avoid false sharing */
    chunk = (chunk + 7) & ~((int64_t)7);
    if (chunk > K) chunk = K;

    #pragma omp parallel for schedule(static)
    for (int64_t rb = 0; rb < K; rb += chunk) {
        int64_t re = rb + chunk;
        if (re > K) re = K;
        for (int64_t base = K; base < LEN_1D; base += K) {
            int64_t ib = base + rb;
            if (ib >= LEN_1D) break;
            int64_t ie = base + re;
            if (ie > LEN_1D) ie = LEN_1D;
            for (int64_t i = ib; i < ie; i++) {
                a[i] = 0.75 * a[i - K] + b[i] * c[i];
            }
        }
    }
}
