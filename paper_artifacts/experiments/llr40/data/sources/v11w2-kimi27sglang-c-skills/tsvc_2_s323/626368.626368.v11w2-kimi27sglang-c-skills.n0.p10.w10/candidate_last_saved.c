#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#ifndef PREFIX_THREADS
#define PREFIX_THREADS 8
#endif

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e, const int64_t LEN_1D) {
    if (LEN_1D <= 1) return;

    const int64_t N = LEN_1D - 1;
    const int Tmax = omp_get_max_threads();

    if (Tmax <= 1) {
        for (int64_t i = 1; i < LEN_1D; ++i) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
        return;
    }

    const int K = PREFIX_THREADS;
    size_t scratch_bytes = ((2 * (size_t)K * sizeof(double) + 63) / 64) * 64;
    double *restrict scratch = (double *)aligned_alloc(64, scratch_bytes);
    if (!scratch) return;

    // Region 1: K threads compute block sums and sequential offsets
    #pragma omp parallel num_threads(K)
    {
        const int nt = omp_get_num_threads();
        const int t = omp_get_thread_num();
        const int64_t chunk = N / nt;
        const int64_t rem = N % nt;
        const int64_t start = 1 + t * chunk + (t < rem ? t : rem);
        const int64_t end = start + chunk + (t < rem ? 1 : 0);

        double block_sum = 0.0;
        for (int64_t i = start; i < end; ++i) {
            block_sum = block_sum + c[i] * d[i];
            block_sum = block_sum + c[i] * e[i];
        }
        scratch[t] = block_sum;

        #pragma omp barrier

        #pragma omp single
        {
            scratch[nt] = b[0];
            for (int i = 1; i < nt; ++i) {
                scratch[nt + i] = scratch[nt + i - 1] + scratch[i - 1];
            }
        }
    }

    // Region 2: max threads compute a and b from offsets
    #pragma omp parallel for simd schedule(static)
    for (int64_t t = 0; t < K; ++t) {
        const int64_t chunk = N / K;
        const int64_t rem = N % K;
        const int64_t start = 1 + t * chunk + (t < rem ? t : rem);
        const int64_t end = start + chunk + (t < rem ? 1 : 0);
        double acc = scratch[K + t];
        for (int64_t i = start; i < end; ++i) {
            a[i] = acc + c[i] * d[i];
            acc = a[i] + c[i] * e[i];
            b[i] = acc;
        }
    }

    free(scratch);
}
