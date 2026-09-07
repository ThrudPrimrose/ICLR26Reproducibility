#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict dst, const int64_t *restrict src, const double *restrict w, const double *restrict x, const int64_t E, const int64_t N) {
    int nt = omp_get_max_threads();
    if (nt <= 1) {
        for (int64_t i = 0; i < N; ++i) Lx[i] = 0.0;
        for (int64_t e = 0; e < E; ++e) {
            double f = w[e] * (x[src[e]] - x[dst[e]]);
            Lx[src[e]] += f;
            Lx[dst[e]] -= f;
        }
        return;
    }
    size_t stride = (size_t)N;
    double *restrict tmp = (double *)malloc((size_t)nt * stride * sizeof(double));
    if (!tmp) return;
    memset(tmp, 0, (size_t)nt * stride * sizeof(double));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *restrict loc = tmp + (size_t)tid * stride;
        #pragma omp for schedule(static)
        for (int64_t e = 0; e < E; ++e) {
            double f = w[e] * (x[src[e]] - x[dst[e]]);
            loc[src[e]] += f;
            loc[dst[e]] -= f;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (int t = 0; t < nt; ++t) {
            s += tmp[(size_t)t * stride + i];
        }
        Lx[i] = s;
    }

    free(tmp);
}
