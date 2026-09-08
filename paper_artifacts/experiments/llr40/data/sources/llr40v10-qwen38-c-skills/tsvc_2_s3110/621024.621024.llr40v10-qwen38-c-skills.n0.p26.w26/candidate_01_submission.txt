/* tsvc_2_s3110: find max of LEN_2D x LEN_2D array + first occurrence
 * (row-major, strict 'greater' semantics of the reference),
 * chksum = maxv + xindex + yindex.
 *
 * Parallel form: single flat pass over N = LEN_2D*LEN_2D elements. Each thread
 * keeps a (value, position) candidate and keeps the better one under the total
 * order "larger value wins, tie -> smaller position". That order is commutative
 * and associative, so merging the per-thread candidates afterwards (in any
 * order) reproduces exactly the reference's first-occurrence choice.
 */
#include <stdint.h>
#include <math.h>
#include <omp.h>

typedef struct { double v; int64_t p; } cand_t;

static inline cand_t pick(cand_t a, cand_t b) {
    if (b.v > a.v) return b;
    if (a.v > b.v) return a;
    int na = (a.v < a.v), nb = (b.v < b.v); /* NaN never wins */
    if (na != nb) return na ? b : a;
    return (a.p <= b.p) ? a : b;
}

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t N = LEN_2D * LEN_2D;
    const int64_t L = LEN_2D;

    if (N >= 32768) {
        const int nt = omp_get_max_threads();
        cand_t partials[1024];
        const int T = (nt > 1024) ? 1024 : nt;
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            cand_t c;
            c.v = -INFINITY;
            c.p = INT64_MAX;
#pragma omp for schedule(static)
            for (int64_t k = 0; k < N; ++k) {
                cand_t e;
                e.v = aa[k];
                e.p = k;
                c = pick(c, e);
            }
            partials[tid] = c;
        }
        cand_t c = partials[0];
        for (int t = 1; t < T; ++t) c = pick(c, partials[t]);
        const int64_t xindex = c.p / L;
        const int64_t yindex = c.p % L;
        bb[0] = c.v + (double)xindex + (double)yindex;
    } else {
        cand_t c;
        c.v = -INFINITY;
        c.p = INT64_MAX;
        for (int64_t k = 0; k < N; ++k) {
            cand_t e;
            e.v = aa[k];
            e.p = k;
            c = pick(c, e);
        }
        const int64_t xindex = c.p / L;
        const int64_t yindex = c.p % L;
        bb[0] = c.v + (double)xindex + (double)yindex;
    }
}
