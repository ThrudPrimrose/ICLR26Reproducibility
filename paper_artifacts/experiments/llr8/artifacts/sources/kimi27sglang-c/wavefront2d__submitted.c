#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <immintrin.h>
#include <omp.h>

static inline void process_block(double *restrict a, int64_t N,
                                 int64_t bi, int64_t bj, int64_t B) {
    int64_t i0 = bi * B; if (i0 < 1) i0 = 1;
    int64_t i1 = (bi + 1) * B - 1; if (i1 >= N) i1 = N - 1;
    int64_t j0 = bj * B; if (j0 < 1) j0 = 1;
    int64_t j1 = (bj + 1) * B - 1; if (j1 >= N) j1 = N - 1;
    for (int64_t i = i0; i <= i1; ++i) {
        double *restrict row = a + i * N;
        const double *restrict rowp = a + (i - 1) * N;
        for (int64_t j = j0; j <= j1; ++j) {
            row[j] = 0.25 * (row[j] + rowp[j] + row[j - 1] + rowp[j - 1]);
        }
    }
}

void wavefront2d_fp64(double *restrict a, const int64_t LEN_2D) {
    int64_t N = LEN_2D;
    if (N <= 512) {
        #pragma omp parallel
        {
            for (int64_t d = 2; d <= 2 * (N - 1); ++d) {
                int64_t i0 = d - (N - 1);
                if (i0 < 1) i0 = 1;
                int64_t i1 = d - 1;
                if (i1 > N - 1) i1 = N - 1;
                #pragma omp for schedule(static,1) nowait
                for (int64_t i = i0; i <= i1; ++i) {
                    int64_t j = d - i;
                    a[i * N + j] = 0.25 * (a[i * N + j] + a[(i - 1) * N + j] +
                                           a[i * N + (j - 1)] + a[(i - 1) * N + (j - 1)]);
                }
                #pragma omp barrier
            }
        }
        return;
    }

    const int64_t B = 104;
    int64_t nb = (N + B - 1) / B;

    int max_threads = omp_get_max_threads();
    if (nb < max_threads) {
        #pragma omp parallel
        {
            const int nt = omp_get_num_threads();
            const int tid = omp_get_thread_num();
            for (int64_t s = 0; s <= 2 * (nb - 1); ++s) {
                int64_t bi0 = s - (nb - 1);
                if (bi0 < 0) bi0 = 0;
                int64_t bi1 = s;
                if (bi1 > nb - 1) bi1 = nb - 1;
                int64_t nblocks = bi1 - bi0 + 1;
                for (int64_t idx = tid; idx < nblocks; idx += nt) {
                    int64_t bi = bi0 + idx;
                    int64_t bj = s - bi;
                    process_block(a, N, bi, bj, B);
                }
                #pragma omp barrier
            }
        }
        return;
    }

    _Atomic unsigned char *done = calloc((size_t)nb * (size_t)nb,
                                         sizeof(_Atomic unsigned char));
    if (!done) return;

    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        int rows = (int)((nb + nt - 1) / nt);
        int r0 = tid * rows;
        int r1 = r0 + rows;
        if (r1 > (int)nb) r1 = (int)nb;

        if (r0 < (int)nb) {
            for (int64_t bj = 0; bj < nb; ++bj) {
                if (r0 > 0) {
                    _Atomic unsigned char *top = &done[(r0 - 1) * nb + bj];
                    while (!atomic_load_explicit(top, memory_order_acquire))
                        _mm_pause();
                }
                for (int bi = r0; bi < r1; ++bi) {
                    process_block(a, N, bi, bj, B);
                }
                atomic_store_explicit(&done[(r1 - 1) * nb + bj], 1, memory_order_release);
            }
        }
    }

    free(done);
}
