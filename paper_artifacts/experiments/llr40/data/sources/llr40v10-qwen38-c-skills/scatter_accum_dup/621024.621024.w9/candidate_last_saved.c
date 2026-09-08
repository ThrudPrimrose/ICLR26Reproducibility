#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Indexed accumulate with conflicting indices: bins[ip[i]] += src[i], ip may repeat.
 *
 * Strategy: LSD radix-sort the (ip, src) pairs by the 27 significant bits of ip
 * (3 stable 9-bit passes, ping-ponging through two workspace buffers), then a
 * single sequential-order pass: equal keys form short contiguous runs whose src
 * values are summed and applied to bins[key] exactly once -- one stream over
 * everything, zero atomics, zero random read-modify-writes.
 *
 * Per-thread bucket counters/positions make each pass stable across threads:
 * every thread owns one contiguous input interval and one contiguous output
 * segment inside each bucket, ordered by thread id == interval order.
 */
#define NCNT 512
#define MAXT 256

static void radix_pass(const int32_t *restrict ip_in, const double *restrict src_in,
                       int32_t *restrict ip_out, double *restrict src_out,
                       const int64_t N, const int64_t shift, const int nt,
                       uint64_t *restrict tables) {
    uint64_t *restrict cnt_t = tables;            /* [nt][NCNT] */
    uint64_t *restrict pos_t = tables + (int64_t)MAXT * NCNT; /* [nt][NCNT] */

    for (int64_t t = 0; t < nt; t++)
        memset(cnt_t + t * NCNT, 0, NCNT * sizeof(uint64_t));

    int64_t chunk = N / nt, rem = N % nt;
    const int32_t *ip = ip_in;
    const double *sv = src_in;

#pragma omp parallel num_threads(nt)
    {
        const int t = omp_get_thread_num();
        const int64_t lo = t * chunk + (t < rem ? t : rem);
        const int64_t hi = (t + 1) * chunk + (t + 1 < rem ? t + 1 : rem);
        uint64_t *ct = cnt_t + (int64_t)t * NCNT;
        for (int64_t i = lo; i < hi; i++)
            ct[((uint64_t)(uint32_t)ip[i] >> (uint64_t)shift) & (NCNT - 1)]++;
    }

    uint64_t off[NCNT + 1];
    off[0] = 0;
    for (int k = 0; k < NCNT; k++) {
        uint64_t s = 0;
        for (int t = 0; t < nt; t++) s += cnt_t[(int64_t)t * NCNT + k];
        off[k + 1] = off[k] + s;
    }
    for (int k = 0; k < NCNT; k++) {
        uint64_t r = off[k];
        for (int t = 0; t < nt; t++) {
            pos_t[(int64_t)t * NCNT + k] = r;
            r += cnt_t[(int64_t)t * NCNT + k];
        }
    }

#pragma omp parallel num_threads(nt)
    {
        const int t = omp_get_thread_num();
        const int64_t lo = t * chunk + (t < rem ? t : rem);
        const int64_t hi = (t + 1) * chunk + (t + 1 < rem ? t + 1 : rem);
        uint64_t *pp = pos_t + (int64_t)t * NCNT;
        for (int64_t i = lo; i < hi; i++) {
            const uint64_t k = ((uint64_t)(uint32_t)ip[i] >> (uint64_t)shift) & (NCNT - 1);
            const int64_t p = (int64_t)(pp[k]++);
            ip_out[p] = ip[i];
            src_out[p] = sv[i];
        }
    }
}

void scatter_accum_dup_fp64(double *restrict bins,
                            const int32_t *restrict ip,
                            const double *restrict src,
                            const int64_t LEN_1D,
                            uint8_t *restrict workspace,
                            const int64_t workspace_size) {
    const int64_t N = LEN_1D;
    if (N <= 0) return;

    const int64_t need = 24 * N + (int64_t)MAXT * NCNT * 2 * (int64_t)sizeof(uint64_t) + 64;
    if (workspace == NULL || workspace_size < need) {
        /* Fallback (no scratch granted): safe atomic accumulate. */
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N; i++) {
#pragma omp atomic update
            bins[ip[i]] += src[i];
        }
        return;
    }

    /* cap nt to the threads that will actually be used: bound the team size */
    int32_t *restrict ipB = (int32_t *)(workspace);
    double  *restrict srcB = (double *)(workspace + 4 * N);
    int32_t *restrict ipC = (int32_t *)(workspace + 12 * N);
    double  *restrict srcC = (double *)(workspace + 16 * N);
    uint64_t *restrict tables = (uint64_t *)(workspace + 24 * N);

    const int64_t npass = (N >= (1LL << 27)) ? 4 : 3;
    int nt = omp_get_max_threads();
    if (nt < 1) nt = 1;
    if (nt > MAXT) {
        nt = MAXT; /* cap: any extra OMP thread stays idle at the barriers */
    }

    const int32_t *cur_ip = ip;
    const double *cur_src = src;
    for (int64_t p = 0; p < npass; p++) {
        int32_t *nxt_ip;
        double *nxt_src;
        if (p & 1) { nxt_ip = ipC; nxt_src = srcC; }
        else       { nxt_ip = ipB; nxt_src = srcB; }
        radix_pass(cur_ip, cur_src, nxt_ip, nxt_src, N, 9 * p, nt, tables);
        cur_ip = nxt_ip;
        cur_src = nxt_src;
    }

    /* Sorted: equal keys contiguous, keys ascending -> bins touched in order. */
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; i++) {
        if (i != 0 && cur_ip[i] == cur_ip[i - 1]) continue;
        double s = 0.0;
        int64_t k = i;
        while (k < N && cur_ip[k] == cur_ip[i]) {
            s += cur_src[k];
            k++;
        }
        bins[cur_ip[i]] = bins[cur_ip[i]] + s;
    }
}
