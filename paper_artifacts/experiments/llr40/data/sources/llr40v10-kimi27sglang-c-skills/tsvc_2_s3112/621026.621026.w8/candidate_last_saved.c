#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    const int64_t n = LEN_1D;

    int max_threads = omp_get_max_threads();
    if (max_threads < 1) max_threads = 1;

    const int64_t min_chunk = 1024;
    int64_t usable_threads = (n + min_chunk - 1) / min_chunk;
    if (usable_threads < 1) usable_threads = 1;
    if (max_threads > (int)usable_threads) max_threads = (int)usable_threads;

    if (max_threads == 1) {
        double sum = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            sum += a[i];
            b[i] = sum;
        }
        return;
    }

    double *restrict totals = (double *)aligned_alloc(64, sizeof(double) * (size_t)max_threads);
    double *restrict offsets = (double *)aligned_alloc(64, sizeof(double) * (size_t)max_threads);

    const int64_t chunk = (n + max_threads - 1) / max_threads;

    #pragma omp parallel num_threads(max_threads)
    {
        const int t = omp_get_thread_num();
        const int64_t start = (int64_t)t * chunk;
        int64_t end = start + chunk;
        if (end > n) end = n;

        double s = 0.0;
        for (int64_t i = start; i < end; ++i) {
            s += a[i];
        }
        totals[t] = s;
    }

    offsets[0] = 0.0;
    for (int t = 1; t < max_threads; ++t) {
        offsets[t] = offsets[t - 1] + totals[t - 1];
    }

    #pragma omp parallel num_threads(max_threads)
    {
        const int t = omp_get_thread_num();
        const int64_t start = (int64_t)t * chunk;
        int64_t end = start + chunk;
        if (end > n) end = n;

        double s = offsets[t];
        for (int64_t i = start; i < end; ++i) {
            s += a[i];
            b[i] = s;
        }
    }

    free(totals);
    free(offsets);
}
