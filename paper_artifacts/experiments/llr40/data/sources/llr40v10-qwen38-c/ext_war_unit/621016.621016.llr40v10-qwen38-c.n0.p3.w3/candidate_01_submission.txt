/* TSVC tsvc_2_5 ext_war_unit, fp64: a[i] = a[i+1] + b[i], i in [0, N-2].
 *
 * The loop has only a distance-1 anti-dependence on a: every READ of a[k]
 * (by iteration k-1) must happen before the WRITE of a[k] (by iteration k),
 * and nothing reads a value after it is written.  Hence all reads see the
 * ORIGINAL contents of a, and the loop is parallelizable:
 *   phase 1 (pre-read): thread t, owning block [lo_t, hi_t) of i-indices,
 *           captures r_t = a[hi_t] (the one element it reads outside its own
 *           block; a[N-1] is never written so this is always safe, and no
 *           write has happened yet anyway);
 *   barrier;
 *   phase 2 (forward sweep): each thread runs i = lo_t .. hi_t-1 in FORWARD
 *           order; within a block the forward order guarantees read(i) before
 *           write(i+1), and the single cross-block read uses the pre-read r_t.
 * Single pass, 3*N double-words of traffic, zero extra memory.
 */
#include <stdint.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t N) {
    if (N <= 1) return;
    const int64_t n = N - 1; /* iterations: i = 0 .. n-1 */

    /* small inputs: plain vectorizable loop, no OpenMP region overhead */
    if (n < 65536) {
        for (int64_t i = 0; i < n; ++i) a[i] = a[i + 1] + b[i];
        return;
    }

    #pragma omp parallel
    {
        const int nt  = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t lo = n * tid / nt;
        const int64_t hi = n * (tid + 1) / nt;

        double r = a[hi];               /* phase 1: pre-read right boundary */
        #pragma omp barrier
        /* phase 2: forward sweep */
        for (int64_t i = lo; i + 1 < hi; ++i) a[i] = a[i + 1] + b[i];
        if (hi > lo) a[hi - 1] = r + b[hi - 1];
    }
}
