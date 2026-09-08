#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

/* Segmented dot product: out[s] = sum_{e in [row_ptr[s], row_ptr[s+1])} val[e] * w[e].
 *
 * Segments are ragged (heavy-tailed lengths), so the outer loop is NOT threaded
 * directly: thread t instead owns the entry range [total*t/T, total*(t+1)/T),
 * which is exactly equal work for every thread. Segments fully inside the range
 * are written directly; segments straddling a range boundary yield at most two
 * partials per thread, combined serially after the barrier. Zero-length segments
 * sitting exactly on a boundary are written as 0.0 by the owning thread's gap fill. */

static inline int64_t lower_bound_ge(const int64_t *rp, int64_t n, int64_t x)
{
    /* first p in [0, n) with rp[p] >= x; rp non-decreasing */
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t m = (lo + hi) >> 1;
        if (rp[m] >= x) hi = m; else lo = m + 1;
    }
    return lo;
}

static inline int64_t lower_bound_gt(const int64_t *rp, int64_t n, int64_t x)
{
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t m = (lo + hi) >> 1;
        if (rp[m] > x) hi = m; else lo = m + 1;
    }
    return lo;
}

void segment_reduce_ragged_fp64(double *restrict out, const int64_t *restrict row_ptr,
                                const double *restrict val, const double *restrict w,
                                const int64_t NSEG, uint8_t *workspace, int64_t workspace_size)
{
    if (NSEG <= 0) return;
    const int64_t total = row_ptr[NSEG];
    if (total <= 0) {
        for (int64_t s = 0; s < NSEG; s++) out[s] = 0.0;
        return;
    }

    const int T_max = omp_get_max_threads() > 0 ? omp_get_max_threads() : 1;
    int64_t *pseg; double *psum;
    int from_arena = 0;
    if (workspace && workspace_size >= 32 * T_max) {
        pseg = (int64_t *)workspace;
        psum = (double *)(workspace + 16 * T_max);
        from_arena = 1;
    } else {
        pseg = (int64_t *)malloc(16 * T_max);
        psum = (double *)malloc(16 * T_max);
    }
    for (int t = 0; t < T_max; t++) { pseg[2*t] = -1; pseg[2*t+1] = -1; }

    #pragma omp parallel
    {
        const int t = omp_get_thread_num();
        const int T = omp_get_num_threads();
        const int64_t lo = (total * (int64_t)t) / T;
        const int64_t hi = (total * (int64_t)(t + 1)) / T;
        if (lo < hi) {
            const int64_t n2 = NSEG + 1;
            const int64_t p_eq = lower_bound_ge(row_ptr, n2, lo); /* first rp >= lo */
            const int64_t p_lo = lower_bound_gt(row_ptr, n2, lo); /* first rp > lo  */
            const int64_t p_hi = lower_bound_ge(row_ptr, n2, hi); /* first rp >= hi */
            const int64_t q    = lower_bound_gt(row_ptr, n2, hi); /* first rp > hi  */

            /* zero-length segments sitting exactly on boundary lo: s in [p_eq, p_lo-2] */
            for (int64_t s = p_eq; s < p_lo - 1; s++) out[s] = 0.0;
            /* zero-length segments sitting exactly on boundary hi: s in [p_hi, q-2] */
            for (int64_t s = p_hi; s < q - 1; s++) out[s] = 0.0;

            const int64_t s0 = p_lo - 1; /* segment containing entry lo   */
            const int64_t s1 = p_hi - 1; /* segment containing entry hi-1 */

            int64_t psegs[2]; double psums[2];
            int64_t npt = 0;
            for (int64_t s = s0; s <= s1; s++) {
                const int64_t a = row_ptr[s] > lo ? row_ptr[s] : lo;
                const int64_t b = row_ptr[s+1] < hi ? row_ptr[s+1] : hi;
                double sum = 0.0;
                #pragma omp simd
                for (int64_t e = a; e < b; e++)
                    sum += val[e] * w[e];
                if (a == row_ptr[s] && b == row_ptr[s+1]) {
                    out[s] = sum;
                } else {
                    psegs[npt] = s; psums[npt] = sum; npt++;
                }
            }
            for (int64_t k = 0; k < npt; k++) {
                pseg[2*t + k] = psegs[k];
                psum[2*t + k] = psums[k];
            }
        }
    }

    /* combine boundary partials: a straddled segment was never written outright */
    for (int t = 0; t < T_max; t++)
        for (int k = 0; k < 2; k++)
            if (pseg[2*t + k] >= 0) out[pseg[2*t + k]] = 0.0;
    for (int t = 0; t < T_max; t++)
        for (int k = 0; k < 2; k++)
            if (pseg[2*t + k] >= 0) out[pseg[2*t + k]] += psum[2*t + k];

    if (!from_arena) {
        free(pseg);
        free(psum);
    }
}
