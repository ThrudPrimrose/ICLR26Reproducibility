#include <stdint.h>
#include <omp.h>

void tsvc_2_s2233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {

    /* Split the two recurrences and parallelize each over its independent dimension.
       The outer loop for aa carries the recurrence (j), so j is sequential and the
       unit-stride i dimension is parallel.  The outer loop for bb carries the
       recurrence (i), so i is sequential and the unit-stride j dimension is parallel.
       Each thread gets a contiguous, cache-line aligned slab. */
    #pragma omp parallel
    {
        const int nt  = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t L = LEN_2D;
        const int64_t n = L - 8;
        int64_t chunk = (n + nt - 1) / nt;
        chunk = (chunk + 7) & ~7;               /* multiple of 8 doubles (cache line) */

        /* aa: parallel over unit-stride columns i, walk rows j outermost. */
        int64_t istart = 8 + tid * chunk;
        int64_t iend   = istart + chunk;
        if (istart > L) istart = L;
        if (iend   > L) iend   = L;

        for (int64_t j = 8; j < L; ++j) {
            for (int64_t i = istart; i < iend; ++i) {
                aa[j * L + i] = aa[(j - 1) * L + i] + cc[j * L + i];
            }
        }

        /* bb: parallel over unit-stride columns j, walk rows i outermost. */
        int64_t jstart = 8 + tid * chunk;
        int64_t jend   = jstart + chunk;
        if (jstart > L) jstart = L;
        if (jend   > L) jend   = L;

        for (int64_t i = 8; i < L; ++i) {
            for (int64_t j = jstart; j < jend; ++j) {
                bb[i * L + j] = bb[(i - 1) * L + j] + cc[i * L + j];
            }
        }
    }
}
