#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <omp.h>

#ifndef BLOCK
#define BLOCK 64
#endif

static void wavefront2d_serial(double *restrict a, int64_t n) {
    for (int64_t i = 1; i < n; ++i) {
        double *restrict row = a + i * n;
        const double *restrict rowm = a + (i - 1) * n;
        for (int64_t j = 1; j < n; ++j) {
            row[j] = 0.25 * (row[j] + rowm[j] + row[j - 1] + rowm[j - 1]);
        }
    }
}

void wavefront2d_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    if (n < 256) {
        wavefront2d_serial(a, n);
        return;
    }

    const int64_t B = BLOCK;
    const int64_t nt = (n + B - 1) / B;

    atomic_uchar *restrict ready = (atomic_uchar *)calloc((size_t)(nt * nt), sizeof(atomic_uchar));
    if (!ready) {
        wavefront2d_serial(a, n);
        return;
    }
    for (int64_t bi = 0; bi < nt; ++bi) {
        atomic_store_explicit(&ready[bi * nt], 1, memory_order_relaxed);
    }

#pragma omp parallel default(none) shared(a, n, B, nt, ready)
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t cols_per = (nt + nthreads - 1) / nthreads;
        const int64_t bj_start = (int64_t)tid * cols_per;
        const int64_t bj_end = (bj_start + cols_per < nt) ? bj_start + cols_per : nt;

        for (int64_t bi = 0; bi < nt; ++bi) {
            for (int64_t bj = bj_start; bj < bj_end; ++bj) {
                if (bj > 0) {
                    while (atomic_load_explicit(&ready[bi * nt + (bj - 1)], memory_order_acquire) == 0) {
                        // spin
                    }
                }
                const int64_t i0 = bi * B;
                const int64_t i1 = (bi + 1) * B;
                const int64_t istart = (i0 < 1) ? 1 : i0;
                const int64_t iend = (i1 < n) ? i1 : n;
                const int64_t j0 = bj * B;
                const int64_t j1 = (bj + 1) * B;
                const int64_t jstart = (j0 < 1) ? 1 : j0;
                const int64_t jend = (j1 < n) ? j1 : n;
                for (int64_t i = istart; i < iend; ++i) {
                    double *restrict row = a + i * n;
                    const double *restrict rowm = a + (i - 1) * n;
                    for (int64_t j = jstart; j < jend; ++j) {
                        row[j] = 0.25 * (row[j] + rowm[j] + row[j - 1] + rowm[j - 1]);
                    }
                }
                atomic_store_explicit(&ready[bi * nt + bj], 1, memory_order_release);
            }
        }
    }

    free(ready);
}
