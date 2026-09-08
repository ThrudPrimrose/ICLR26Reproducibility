#include <stdint.h>
#include <omp.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 4096) {
        double sum = 0.0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            sum += a[i];
            b[i] = sum;
        }
        return;
    }

    int nthreads = 4;
    int64_t chunk = (LEN_1D + nthreads - 1) / nthreads;

    long double block_sums[nthreads];

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int t = 0; t < nthreads; ++t) {
        int64_t start = t * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        long double sum = 0.0L;
        for (int64_t i = start; i < end; ++i) {
            sum += a[i];
        }
        block_sums[t] = sum;
    }

    long double acc = 0.0L;
    for (int t = 0; t < nthreads; ++t) {
        long double tmp = block_sums[t];
        block_sums[t] = acc;
        acc += tmp;
    }

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int t = 0; t < nthreads; ++t) {
        long double sum = block_sums[t];
        int64_t start = t * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        for (int64_t i = start; i < end; ++i) {
            sum += a[i];
            b[i] = (double)sum;
        }
    }
}
