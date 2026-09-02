#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *restrict aa,
                      const double *restrict bb,
                      const double *restrict cc,
                      int64_t LEN_2D,
                      void *restrict workspace,
                      int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    int64_t n = LEN_2D;
    if (n <= 1) return;

    uint8_t *restrict mask = (uint8_t *)malloc((size_t)n * sizeof(uint8_t));
    if (!mask) return;

    for (int64_t i = 0; i < n; ++i) {
        mask[i] = (aa[i] > 0.0) ? 1 : 0;
    }

#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t chunk = (n + nthreads - 1) / nthreads;
        int64_t i0 = tid * chunk;
        int64_t i1 = i0 + chunk;
        if (i1 > n) i1 = n;

        if (i0 < i1) {
            for (int64_t j = 1; j < n; ++j) {
                const double *restrict row_prev = aa + (j - 1) * n;
                const double *restrict row_b = bb + j * n;
                const double *restrict row_c = cc + j * n;
                double *restrict row_a = aa + j * n;
#pragma omp simd
                for (int64_t i = i0; i < i1; ++i) {
                    if (mask[i]) {
                        row_a[i] = row_prev[i] + row_b[i] * row_c[i];
                    }
                }
            }
        }
    }

    free(mask);
}
