#include <stdint.h>
#include <omp.h>

void tsvc_2_s1244_fp64(double *restrict a, const double *restrict b, const double *restrict c, double *restrict d,
                       const int64_t LEN_1D) {
    const int64_t N = LEN_1D - 1;
    if (N <= 0) return;

    const int64_t TILE = 16384;

#pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t base = (N * (int64_t)tid) / (int64_t)nthreads;
        const int64_t limit = (N * (int64_t)(tid + 1)) / (int64_t)nthreads;
        double boundary = 0.0;

        if (limit > base) {
            boundary = a[limit];
        }

#pragma omp barrier

        if (limit > base) {
            d[limit - 1] = boundary;
        }

        for (int64_t tstart = base; tstart < limit; tstart += TILE) {
            int64_t tend = tstart + TILE;
            if (tend > limit) tend = limit;

            int64_t p1_end = (tend == limit) ? (limit - 1) : tend;
            for (int64_t i = tstart; i < p1_end; i++) {
                d[i] = a[i + 1];
            }

            for (int64_t i = tstart; i < tend; i++) {
                double t = b[i] + c[i] * c[i] + b[i] * b[i] + c[i];
                a[i] = t;
                d[i] = t + d[i];
            }
        }
    }
}
