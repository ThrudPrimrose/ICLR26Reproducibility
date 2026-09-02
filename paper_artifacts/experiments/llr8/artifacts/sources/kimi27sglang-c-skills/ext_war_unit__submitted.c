#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void ext_war_unit_fp64(double *restrict a, double *restrict b, int64_t LEN_1D,
                       uint8_t *restrict workspace, int64_t workspace_bytes) {
    int64_t n = LEN_1D;
    if (n <= 1) return;

    double *restrict c;
    if (workspace_bytes >= n * (int64_t)sizeof(double)) {
        c = (double *)workspace;
    } else {
        c = (double *)aligned_alloc(64, ((size_t)n * sizeof(double) + 63) & ~63ULL);
    }

#pragma omp parallel
    {
        int64_t nt = omp_get_num_threads();
        int64_t tid = omp_get_thread_num();
        int64_t chunk = (n + nt - 1) / nt;
        int64_t lo = tid * chunk;
        int64_t hi = lo + chunk;
        if (hi > n) hi = n;
        if (lo < hi) {
            memcpy(c + lo, a + lo, (size_t)(hi - lo) * sizeof(double));
        }
#pragma omp barrier
#pragma omp for simd schedule(static)
        for (int64_t i = 0; i < n - 1; ++i) {
            a[i] = c[i + 1] + b[i];
        }
    }

    if (c != (double *)workspace) free(c);
}
