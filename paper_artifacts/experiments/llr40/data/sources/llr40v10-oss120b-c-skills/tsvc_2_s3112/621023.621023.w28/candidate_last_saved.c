#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;
    // For small arrays, use a simple serial scan to avoid parallel overhead.
    if (LEN_1D < 500000) {
        double sum = 0.0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            sum += a[i];
            b[i] = sum;
        }
        return;
    }
    int max_threads = omp_get_max_threads();
    if (max_threads > 8) max_threads = 8; // cap to limited threads
    double block_sum[256];
    double offset[256];
    #pragma omp parallel num_threads(max_threads)
    {
        int tid = omp_get_thread_num();
        int64_t chunk = (LEN_1D + max_threads - 1) / max_threads; // ceil division
        int64_t start = tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;
        double sum = 0.0;
        for (int64_t i = start; i < end; ++i) {
            sum += a[i];
        }
        block_sum[tid] = sum;
        #pragma omp barrier
        #pragma omp single
        {
            double acc = 0.0;
            double c = 0.0;
            for (int i = 0; i < max_threads; ++i) {
                offset[i] = acc;
                double y = block_sum[i] - c;
                double t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }
        }
        #pragma omp barrier
        double cur = offset[tid];
        for (int64_t i = start; i < end; ++i) {
            cur += a[i];
            b[i] = cur;
        }
    }
}
