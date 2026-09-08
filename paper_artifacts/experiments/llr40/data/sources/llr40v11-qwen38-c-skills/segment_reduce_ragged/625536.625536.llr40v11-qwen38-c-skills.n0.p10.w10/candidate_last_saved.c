#include <stdint.h>
#include <stddef.h>
#include <omp.h>
#include <stdio.h>

/* Segmented dot product over ragged CSR-style structure.
 *
 * Each segment s covers flat indices [row_ptr[s], row_ptr[s+1]) of val/w,
 * and out[s] = sum_{e in segment} val[e]*w[e].
 *
 * Parallel plan: assign each segment to exactly one thread by its midpoint:
 * thread t owns segment s iff lo_t <= (row_ptr[s]+row_ptr[s+1])/2 < hi_t with
 * lo_t = total*t/T, hi_t = total*(t+1)/T.  Midpoints are nondecreasing, so
 * every thread's segment set is a contiguous range found by one binary
 * search; the partition is balanced to within one segment's length and no
 * segment straddles two threads (no atomics, no partials to merge).
 */
void segment_reduce_ragged_fp64(double *restrict out,
                                const int64_t *restrict row_ptr,
                                const double *restrict val,
                                const double *restrict w,
                                const int64_t NSEG)
{
    if (NSEG <= 0) return;
    const int64_t total = row_ptr[NSEG];

    if (total == 0) {
        for (int64_t s = 0; s < NSEG; ++s) out[s] = 0.0;
        return;
    }

    const int nt = omp_get_max_threads();

    if (nt <= 1 || total < (1 << 18)) {
        /* small: thread fork would cost more than the work */
        fprintf(stdout, "DBG serial nt=%d NSEG=%lld total=%lld\n", nt, (long long)NSEG, (long long)total); fflush(stdout);
        for (int64_t s = 0; s < NSEG; ++s) {
            const int64_t e0 = row_ptr[s];
            const int64_t e1 = row_ptr[s + 1];
            double acc = 0.0;
            #pragma omp simd
            for (int64_t e = e0; e < e1; ++e) acc += val[e] * w[e];
            out[s] = acc;
        }
        return;
    }

    #pragma omp parallel
    {
        const int64_t T = (int64_t)omp_get_num_threads();
        const int64_t tid = (int64_t)omp_get_thread_num();
        if (tid == 0) { fprintf(stdout, "DBG par nt=%d NSEG=%lld total=%lld\n", (int)T, (long long)NSEG, (long long)total); fflush(stdout); }
        const int64_t lo2 = 2 * total * tid / T;
        const int64_t hi2 = 2 * total * (tid + 1) / T;

        /* first segment with midpoint (doubled) >= lo2 */
        int64_t b_lo = 0, b_hi = NSEG;
        while (b_lo < b_hi) {
            const int64_t mid = b_lo + (b_hi - b_lo) / 2;
            if (row_ptr[mid] + row_ptr[mid + 1] >= lo2) b_hi = mid;
            else b_lo = mid + 1;
        }

        for (int64_t s = b_lo; s < NSEG; ++s) {
            if (row_ptr[s] + row_ptr[s + 1] >= hi2) break;
            const int64_t e0 = row_ptr[s];
            const int64_t e1 = row_ptr[s + 1];
            double acc = 0.0;
            #pragma omp simd
            for (int64_t e = e0; e < e1; ++e) acc += val[e] * w[e];
            out[s] = acc;
        }
    }
}
