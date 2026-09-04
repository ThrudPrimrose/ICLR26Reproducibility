/* Stream compaction: pack src[i]*weight[i] for src[i] > 0, publish count.
 * Two-pass: per-chunk survivor counts + tiny prefix sum, then parallel scatter. */
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void compact_threshold_pack_fp64(
    int64_t *restrict out_count,
    double *restrict packed,
    const double *restrict src,
    const double *restrict weight,
    const int64_t LEN_1D)
{
    const int nt = omp_get_max_threads();
    int64_t *cc = (int64_t *)malloc(sizeof(int64_t) * (size_t)nt);
    if (!cc) return;

    const int64_t cs = (LEN_1D + nt - 1) / nt;

    /* pass 1: branchless, vectorizable per-chunk survivor counts */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int64_t lo = (int64_t)t * cs;
        int64_t hi = lo + cs;
        if (hi > LEN_1D) hi = LEN_1D;
        int64_t c = 0;
        for (int64_t i = lo; i < hi; i++) c += (src[i] > 0.0);
        cc[t] = c;
    }

    /* prefix sum over chunk counts (tiny) */
    int64_t total = 0;
    for (int t = 0; t < nt; t++) {
        int64_t old = cc[t];
        cc[t] = total;          /* cc[t] reused as the chunk's write base */
        total += old;
    }

    /* pass 2: parallel scatter, sequential cursor within each chunk */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int64_t lo = (int64_t)t * cs;
        int64_t hi = lo + cs;
        if (hi > LEN_1D) hi = LEN_1D;
        int64_t cur = cc[t];
        for (int64_t i = lo; i < hi; i++) {
            if (src[i] > 0.0) {
                packed[cur] = src[i] * weight[i];
                cur++;
            }
        }
    }

    out_count[0] = total;
    free(cc);
}
