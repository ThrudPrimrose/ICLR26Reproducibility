#include <stdint.h>
#include <stdio.h>
#include <omp.h>

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
    static int printed = 0;
    if (!printed) {
        printed = 1;
        printf("INFO N=%lld TSTEPS=%lld max_threads=%d\n",
               (long long)N, (long long)TSTEPS, (int)omp_get_max_threads());
        fflush(stdout);
    }

    for (int64_t t = 0; t < TSTEPS; ++t) {
        #pragma omp parallel for schedule(static)
        for (int64_t si0 = 1; si0 < N - 1; ++si0) {
            const double *cur  = A + si0 * N;
            const double *up   = cur - N;
            const double *down = cur + N;
            double *dst        = B + si0 * N;
            for (int64_t si1 = 1; si1 < N - 1; ++si1) {
                dst[si1] = 0.2 * (((cur[si1] + cur[si1 - 1]) + cur[si1 + 1]
                                   + down[si1]) + up[si1]);
            }
        }
        #pragma omp parallel for schedule(static)
        for (int64_t si0 = 1; si0 < N - 1; ++si0) {
            const double *cur  = B + si0 * N;
            const double *up   = cur - N;
            const double *down = cur + N;
            double *dst        = A + si0 * N;
            for (int64_t si1 = 1; si1 < N - 1; ++si1) {
                dst[si1] = 0.2 * (((cur[si1] + cur[si1 - 1]) + cur[si1 + 1]
                                   + down[si1]) + up[si1]);
            }
        }
    }
}
