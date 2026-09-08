#include <stdint.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b,
                       const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
    const int64_t n = LEN_1D - 1;
    if (n <= 0) return;

    /* The reference loop uses the freshly-written a[i] and the *original* a[i+1].
       Fuse both statements into one forward pass: read a[i+1] before any thread
       overwrites it.  Each thread snapshots the original value at its chunk's
       right edge, then a barrier guarantees the snapshot happens before writes. */

    const int max_threads = omp_get_max_threads();
    const int nthreads = (n > 4096 && max_threads > 1)
                         ? (max_threads < 8 ? max_threads : 8)
                         : 1;

    #pragma omp parallel if(n > 4096) num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const int64_t chunk = (n + nt - 1) / nt;
        const int64_t lo = tid * chunk;
        int64_t hi = lo + chunk;
        if (hi > n) hi = n;

        double right = 0.0;
        if (lo < hi) {
            right = a[hi];
        }
        #pragma omp barrier

        int64_t i = lo;
        for (; i + 1 < hi; ++i) {
            const double bi = b[i];
            const double ci = c[i];
            const double t = bi * bi + bi + ci * ci + ci;
            d[i] = t + a[i + 1];
            a[i] = t;
        }
        if (i < hi) {
            const double bi = b[i];
            const double ci = c[i];
            const double t = bi * bi + bi + ci * ci + ci;
            d[i] = t + right;
            a[i] = t;
        }
    }
}
