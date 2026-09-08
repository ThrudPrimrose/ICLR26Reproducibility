#include <stdint.h>
#include <omp.h>

/*
 * Parallelized version of wf_diff_skew using OpenMP.
 * The inner loop over j is independent for a given i, so we parallelize it.
 */

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    if (LEN_2D < 2) return; // nothing to do
    #pragma omp parallel default(none) shared(a, LEN_2D)
    {
        for (int64_t i = 1; i < LEN_2D; ++i) {
            #pragma omp for schedule(static)
            for (int64_t j = 0; j < LEN_2D - 1; ++j) {
                const int64_t idx      = i * LEN_2D + j;
                const int64_t idx_up   = (i - 1) * LEN_2D + j;
                const int64_t idx_up_r = (i - 1) * LEN_2D + (j + 1);
                a[idx] = a[idx] + a[idx_up] + a[idx_up_r];
            }
        }
    }
}
