#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    const int nt = omp_get_max_threads();
    if (LEN_1D < 4096 || nt <= 1) {
        double sum = 0.0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            sum += a[i];
            b[i] = sum;
        }
        return;
    }

    double *restrict totals = (double *)malloc((size_t)nt * sizeof(double));
    if (totals == NULL) {
        double sum = 0.0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            sum += a[i];
            b[i] = sum;
        }
        return;
    }

    #pragma omp parallel
    {
        const int actual = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t chunk = (LEN_1D + actual - 1) / actual;
        int64_t start = (int64_t)tid * chunk;
        if (start > LEN_1D) start = LEN_1D;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;

        double local = 0.0;
        for (int64_t i = start; i < end; ++i) {
            local += a[i];
        }
        totals[tid] = local;

        #pragma omp barrier
        if (tid == 0) {
            double acc = 0.0;
            for (int t = 0; t < actual; ++t) {
                double v = totals[t];
                totals[t] = acc;
                acc += v;
            }
        }
        #pragma omp barrier

        double prefix = totals[tid];
        for (int64_t i = start; i < end; ++i) {
            prefix += a[i];
            b[i] = prefix;
        }
    }

    free(totals);
}
