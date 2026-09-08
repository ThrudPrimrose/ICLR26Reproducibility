/* Stream compaction (compact_threshold_pack) -- GPU via OpenMP target offload.
 *
 * ABI (judge-generated binding, symbol compact_threshold_pack_fp64):
 *   void compact_threshold_pack_fp64(int64_t *out_count, double *packed,
 *                                    const double *src, const double *weight,
 *                                    int64_t LEN_1D,
 *                                    uint8_t *workspace, int64_t workspace_bytes);
 *
 * Pack packed[n++] = src[i]*weight[i] for every src[i] > 0.0 (source order),
 * publish out_count[0] = n.  Explicit-map offload model (HSA_XNACK=0):
 * all host buffers cross via map clauses inside the timed section.
 *
 * Algorithm (3 device phases, team = one workgroup):
 *   A.  every thread counts survivors in its contiguous sub-block of its team
 *       chunk; per-thread counts land in device scratch s[team][thread].
 *   B.  one thread prefixes the 608 team totals into toff[team] and publishes
 *       out_count.
 *   C.  every thread re-reads its sub-block and scatters survivors to
 *       packed[toff[team] + sum_{j<tid} s[team][j] + local_run].
 * HBM traffic: src read twice, weight once, packed written once.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>

#define NTEAMS   608
#define NTHREADS 256

void compact_threshold_pack_fp64(int64_t *out_count,
                                 double *packed,
                                 const double *src,
                                 const double *weight,
                                 const int64_t LEN_1D,
                                 uint8_t *workspace,
                                 int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    const int64_t N = LEN_1D;
    if (N <= 0) { out_count[0] = 0; return; }

    const int64_t team = (N + NTEAMS - 1) / NTEAMS;   /* elements per team */
    int32_t *s_dev = NULL;                            /* [NTEAMS][NTHREADS] */
    int64_t *toff_dev = NULL;                         /* [NTEAMS] offsets   */

#pragma omp target data \
    map(to: src[0:N]) \
    map(to: weight[0:N]) \
    map(to: packed[0:N]) \
    map(tofrom: out_count[0:1]) \
    map(alloc: s_dev[0:NTEAMS * NTHREADS]) \
    map(alloc: toff_dev[0:NTEAMS])
    {
        /* ---- Phase A: per-thread survivor counts ---------------------- */
#pragma omp target teams(num_teams(NTEAMS), num_threads(NTHREADS))
        {
            const int64_t tid = omp_get_thread_num();
            const int64_t t = omp_get_team_num();
            int64_t lo = t * team;
            int64_t hi = lo + team;
            if (hi > N) hi = N;
            int32_t *sb = s_dev + t * NTHREADS;
            int32_t cnt = 0;
#pragma omp parallel for schedule(static) private(cnt)
            for (int64_t i = lo; i < hi; i++) {
                if (src[i] > 0.0) cnt++;
            }
            sb[tid] = cnt;
        }

        /* ---- Phase B: prefix the team totals -------------------------- */
#pragma omp target
        {
            int64_t run = 0;
            for (int64_t t = 0; t < NTEAMS; t++) {
                int32_t tt = 0;
                const int32_t *sb = s_dev + t * NTHREADS;
                for (int64_t j = 0; j < NTHREADS; j++) tt += sb[j];
                toff_dev[t] = run;
                run += tt;
            }
            out_count[0] = run;
        }

        /* ---- Phase C: scatter survivors -------------------------------- */
#pragma omp target teams(num_teams(NTEAMS), num_threads(NTHREADS))
        {
            const int64_t tid = omp_get_thread_num();
            const int64_t t = omp_get_team_num();
            int64_t lo = t * team;
            int64_t hi = lo + team;
            if (hi > N) hi = N;
            const int32_t *sb = s_dev + t * NTHREADS;
            int64_t base = toff_dev[t];
            for (int64_t j = 0; j < tid; j++) base += sb[j];
            int64_t run = 0;
#pragma omp parallel for schedule(static) private(run)
            for (int64_t i = lo; i < hi; i++) {
                const double v = src[i];
                if (v > 0.0) {
                    packed[base + run] = v * weight[i];
                    run++;
                }
            }
        }
    }
}
