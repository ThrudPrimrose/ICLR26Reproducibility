#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
void edge_laplacian_fp64(double *restrict Lx, const int64_t *restrict src,
                         const int64_t *restrict dst, const double *restrict w,
                         const double *restrict x, int64_t E, int64_t N)
{
    int nt = omp_get_max_threads(); if (nt < 1) nt = 1;

    /* small case: serial (cache-resident), 4x unroll for MSL, avoids OMP overhead */
    if (nt == 1 || E < (1 << 20)) {
        memset(Lx, 0, (size_t)N * sizeof(double));
        int64_t h = E >> 2, base = h << 2;
        for (int64_t i0 = 0; i0 < h; ++i0) {
            int64_t i = i0 << 2;
            double f1 = w[i]   * (x[src[i]]   - x[dst[i]]);
            double f2 = w[i+1] * (x[src[i+1]] - x[dst[i+1]]);
            double f3 = w[i+2] * (x[src[i+2]] - x[dst[i+2]]);
            double f4 = w[i+3] * (x[src[i+3]] - x[dst[i+3]]);
            Lx[src[i]]   += f1;  Lx[dst[i]]   -= f1;
            Lx[src[i+1]] += f2;  Lx[dst[i+1]] -= f2;
            Lx[src[i+2]] += f3;  Lx[dst[i+2]] -= f3;
            Lx[src[i+3]] += f4;  Lx[dst[i+3]] -= f4;
        }
        for (int64_t i = base; i < E; ++i) {
            double f = w[i] * (x[src[i]] - x[dst[i]]);
            Lx[src[i]] += f;
            Lx[dst[i]] -= f;
        }
        return;
    }

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int64_t per = (N + nt - 1) / nt;
        int64_t lo = (int64_t)tid * per, hi = lo + per; if (hi > N) hi = N;
        memset(Lx + lo, 0, (size_t)(hi - lo) * sizeof(double));
        #pragma omp barrier
        int64_t h = E >> 2, base = h << 2;
        #pragma omp for schedule(static)
        for (int64_t i0 = 0; i0 < h; ++i0) {
            int64_t i = i0 << 2;
            int64_t i2 = i + 256;
            if (i2 + 3 < E) {
                __builtin_prefetch(x + src[i2], 0, 1);
                __builtin_prefetch(x + dst[i2], 0, 1);
                __builtin_prefetch(x + src[i2+1], 0, 1);
                __builtin_prefetch(x + dst[i2+1], 0, 1);
                __builtin_prefetch(x + src[i2+2], 0, 1);
                __builtin_prefetch(x + dst[i2+2], 0, 1);
                __builtin_prefetch(x + src[i2+3], 0, 1);
                __builtin_prefetch(x + dst[i2+3], 0, 1);
            }
            double f1 = w[i] * (x[src[i]] - x[dst[i]]);
            double f2 = w[i+1] * (x[src[i+1]] - x[dst[i+1]]);
            double f3 = w[i+2] * (x[src[i+2]] - x[dst[i+2]]);
            double f4 = w[i+3] * (x[src[i+3]] - x[dst[i+3]]);
            #pragma omp atomic
            Lx[src[i]] += f1;
            #pragma omp atomic
            Lx[dst[i]] -= f1;
            #pragma omp atomic
            Lx[src[i+1]] += f2;
            #pragma omp atomic
            Lx[dst[i+1]] -= f2;
            #pragma omp atomic
            Lx[src[i+2]] += f3;
            #pragma omp atomic
            Lx[dst[i+2]] -= f3;
            #pragma omp atomic
            Lx[src[i+3]] += f4;
            #pragma omp atomic
            Lx[dst[i+3]] -= f4;
        }
        #pragma omp for schedule(static)
        for (int64_t i = base; i < E; ++i) {
            double f = w[i] * (x[src[i]] - x[dst[i]]);
            #pragma omp atomic
            Lx[src[i]] += f;
            #pragma omp atomic
            Lx[dst[i]] -= f;
        }
    }
}
