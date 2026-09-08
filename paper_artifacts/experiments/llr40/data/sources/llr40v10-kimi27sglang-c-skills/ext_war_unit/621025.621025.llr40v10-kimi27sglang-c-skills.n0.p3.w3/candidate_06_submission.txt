#include <stdint.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D) {
    const int64_t n = LEN_1D - 1;
    if (n <= 0) return;

    const int nt_max = omp_get_max_threads();
    const int64_t min_chunk = 8192;

    /* Tiny inputs: avoid the whole parallel region. */
    if (n < min_chunk) {
        for (int64_t i = 0; i < n; ++i) {
            a[i] = a[i + 1] + b[i];
        }
        return;
    }

    /* Use enough threads to cover the array, but not more than available. */
    int nt = (int)((n + min_chunk - 1) / min_chunk);
    if (nt > nt_max) nt = nt_max;
    if (nt <= 1) {
        for (int64_t i = 0; i < n; ++i) {
            a[i] = a[i + 1] + b[i];
        }
        return;
    }

    /*
     * The loop carries a WAR anti-dependence: iteration i writes a[i] and
     * iteration i-1 reads a[i].  We keep the serial order inside each thread
     * but parallelise across blocks.  Each thread saves the original value at
     * its right boundary before any writes happen, then computes left-to-right
     * using that saved value for the last element of its block.
     */
    #pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        const int64_t chunk = (n + nthreads - 1) / nthreads;
        const int64_t start = (int64_t)tid * chunk;
        int64_t end = (int64_t)(tid + 1) * chunk;
        if (end > n) end = n;

        if (start < end) {
            const double boundary = a[end];
            #pragma omp barrier
            #pragma omp simd
            for (int64_t i = start; i < end - 1; ++i) {
                a[i] = a[i + 1] + b[i];
            }
            a[end - 1] = boundary + b[end - 1];
        }
    }
}
