#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

void scatter_accum_dup_fp64(double *restrict bins, const int32_t *restrict ip,
                            const double *restrict src, const int64_t N) {
    int64_t N4 = N & ~(int64_t)3;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N4; i += 4) {
        int64_t j0 = ip[i], j1 = ip[i+1], j2 = ip[i+2], j3 = ip[i+3];
        double s0 = src[i], s1 = src[i+1], s2 = src[i+2], s3 = src[i+3];
        #pragma omp atomic
        bins[j0] += s0;
        #pragma omp atomic
        bins[j1] += s1;
        #pragma omp atomic
        bins[j2] += s2;
        #pragma omp atomic
        bins[j3] += s3;
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i = N4; i < N; ++i) {
        #pragma omp atomic
        bins[ip[i]] += src[i];
    }
}
