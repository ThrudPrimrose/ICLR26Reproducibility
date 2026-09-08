#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/* Segmented dot product over ragged CSR-style structure.
 *
 * out[s] = sum_{e = row_ptr[s]}^{row_ptr[s+1]-1} val[e] * w[e]
 *
 * Segment lengths are heavy-tailed (lognormal, mean 24), so partitioning
 * SEGMENTS across threads load-imbalances. Instead, partition the ENTRY
 * space [0, row_ptr[NSEG]) evenly: each thread owns a contiguous entry
 * range, exactly 1/NT of the work, and walks the segment boundary vector
 * only to know where its sums end. A segment straddling a thread boundary
 * is split; the two partial sums are combined after a barrier (each
 * boundary is combined by exactly one thread, so no atomics).
 */

/* Lazy-growing per-thread partial buffer (index: thread id). */
static double *g_pl = NULL;
static int g_pl_cap = 0;
static int *g_cl = NULL;

void segment_reduce_ragged_fp64(double *restrict out, const int64_t *restrict row_ptr,
                                const double *restrict val, const double *restrict w,
                                const int64_t NSEG, uint8_t *workspace,
                                const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    if (NSEG <= 0)
        return;
    const int64_t total = row_ptr[NSEG];
    if (total <= 0) { /* every segment is empty */
        #pragma omp parallel for schedule(static)
        for (int64_t s = 0; s < NSEG; ++s)
            out[s] = 0.0;
        return;
    }

    const int maxt = omp_get_max_threads();
    if (g_pl_cap < maxt) {
        free(g_pl);
        free(g_cl);
        g_pl = (double *)malloc((size_t)maxt * sizeof(double));
        g_cl = (int *)malloc((size_t)maxt * sizeof(int));
        g_pl_cap = maxt;
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int T = omp_get_num_threads();
        const int64_t lo = (total * (int64_t)tid) / (int64_t)T;
        const int64_t hi = (total * (int64_t)(tid + 1)) / (int64_t)T;

        double pf = 0.0;    /* partial sum of the segment straddling `lo` (its tail) */
        double plast = 0.0; /* partial sum of the segment straddling `hi` (its head) */
        int cf = 0;
        int64_t s_first = -1;

        if (lo < hi) {
            /* s_first: largest s with row_ptr[s] <= lo (row_ptr non-decreasing) */
            int64_t a64 = 0, b64 = (int64_t)NSEG + 1;
            while (a64 < b64) {
                int64_t m = (a64 + b64) >> 1;
                if (row_ptr[m] <= lo)
                    a64 = m + 1;
                else
                    b64 = m;
            }
            s_first = a64 - 1;
            for (int64_t s = s_first; row_ptr[s] < hi; ++s) {
                int64_t a = row_ptr[s];
                if (a < lo)
                    a = lo;
                int64_t b = row_ptr[s + 1];
                if (b > hi)
                    b = hi;
                double acc = 0.0;
                #pragma omp simd reduction(+:acc)
                for (int64_t i = a; i < b; ++i)
                    acc += val[i] * w[i];
                if (s == s_first && row_ptr[s] < lo) {
                    pf = acc;
                    cf = 1;
                } else if (row_ptr[s + 1] > hi) {
                    plast = acc;
                } else {
                    out[s] = acc;
                }
            }
        }

        g_pl[tid] = plast;
        g_cl[tid] = (lo < hi) ? 1 : 0;
        #pragma omp barrier
        if (cf)
            out[s_first] = g_pl[tid - 1] + pf;
    }
}
