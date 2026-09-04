#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <omp.h>
void tsvc_2_s233_fp64(double *restrict aa, double *restrict bb, const double *restrict cc, const int64_t N) {
    printf("LEN_2D=%lld max_threads=%d num_procs=%d\n", (long long)N, omp_get_max_threads(), omp_get_num_procs());
    static double taa, tbb; static int nt_used;
    #pragma omp parallel num_threads(omp_get_max_threads())
    {
        double t0 = omp_get_wtime();
        #pragma omp for schedule(static)
        for (int64_t i0 = 0; i0 < N - 8; i0 += 16) {
            int64_t m = N - 8 - i0; if (m > 16) m = 16;
            int64_t col = i0 + 8;
            double acc[16];
            for (int64_t k = 0; k < m; ++k) acc[k] = aa[7 * N + col + k];
            for (int64_t j = 8; j < N; ++j) {
                const double *c = cc + j * N + col;
                double *a = aa + j * N + col;
                for (int64_t k = 0; k < m; ++k) { acc[k] += c[k]; a[k] = acc[k]; }
            }
        }
        double t1 = omp_get_wtime();
        #pragma omp for schedule(static)
        for (int64_t j0 = 0; j0 < N - 8; j0 += 8) {
            int64_t m = N - 8 - j0; if (m > 8) m = 8;
            int64_t row = j0 + 8;
            double acc[8];
            for (int64_t k = 0; k < m; ++k) acc[k] = bb[(row + k) * N + 7];
            for (int64_t i = 8; i < N; ++i)
                for (int64_t k = 0; k < m; ++k) {
                    int64_t off = (row + k) * N + i;
                    acc[k] += cc[off];
                    bb[off] = acc[k];
                }
        }
        double t2 = omp_get_wtime();
        #pragma omp single
        { nt_used = omp_get_num_threads(); taa = t1 - t0; tbb = t2 - t1; }
    }
    printf("threads_used=%d part_aa=%.3f ms part_bb=%.3f ms\n", nt_used, taa*1000, tbb*1000);
}
