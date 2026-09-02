#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <omp.h>

typedef struct { _Atomic int v; char pad[64 - sizeof(_Atomic int)]; } flag_t;

void wf_north_west_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    const int64_t n = N - 1;
    if (n <= 0) return;

    const int64_t TS = 64;
    const int64_t nt = (n + TS - 1) / TS;

    int max_thr = omp_get_max_threads();
    if (max_thr > 24) max_thr = 24;
    int P = max_thr;
    if (P > (int)nt) P = (int)nt;
    if (P < 1) P = 1;

    flag_t *flags = (flag_t *)aligned_alloc(64, (size_t)P * nt * sizeof(flag_t));
    memset(flags, 0, (size_t)P * nt * sizeof(flag_t));

    #pragma omp parallel num_threads(P)
    {
        const int p = omp_get_thread_num();
        const int64_t cols = nt / P;
        const int64_t rem = nt % P;
        const int64_t tj0 = p * cols + (p < rem ? p : rem);
        const int64_t ncols = cols + (p < rem ? 1 : 0);

        for (int64_t ti = 0; ti < nt; ++ti) {
            if (p > 0) {
                flag_t *f = &flags[(p - 1) * nt + ti];
                while (atomic_load_explicit(&f->v, memory_order_acquire) == 0)
                    __builtin_ia32_pause();
            }

            for (int64_t tj = tj0; tj < tj0 + ncols; ++tj) {
                const int64_t i0 = 1 + ti * TS;
                const int64_t i1 = (i0 + TS < N) ? (i0 + TS) : N;
                const int64_t j0 = 1 + tj * TS;
                const int64_t j1 = (j0 + TS < N) ? (j0 + TS) : N;
                for (int64_t i = i0; i < i1; ++i) {
                    double *restrict const row = a + i * N;
                    const double *restrict const row_n = a + (i - 1) * N;
                    for (int64_t j = j0; j < j1; ++j) {
                        row[j] = row[j] + row_n[j] + row[j - 1];
                    }
                }
            }

            atomic_store_explicit(&flags[p * nt + ti].v, 1, memory_order_release);
        }
    }

    free(flags);
}
