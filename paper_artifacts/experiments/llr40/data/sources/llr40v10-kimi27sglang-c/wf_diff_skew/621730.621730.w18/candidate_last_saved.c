#include <stdint.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <omp.h>

static inline void row_kernel(double *restrict cur,
                              const double *restrict prev,
                              int64_t M) {
    int64_t j = 0;
    for (; j + 8 <= M; j += 8) {
        __m512d c = _mm512_loadu_pd(cur + j);
        __m512d p = _mm512_loadu_pd(prev + j);
        __m512d p1 = _mm512_loadu_pd(prev + j + 1);
        __m512d s = _mm512_add_pd(c, p);
        s = _mm512_add_pd(s, p1);
        _mm512_storeu_pd(cur + j, s);
    }
    for (; j < M; ++j) {
        cur[j] = cur[j] + prev[j] + prev[j + 1];
    }
}

typedef struct { _Atomic int v; char pad[64 - sizeof(_Atomic int)]; } flag_t;

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    const int64_t M = N - 1;
    const int64_t nrows = N - 1;
    if (nrows <= 0 || M <= 0) return;

    const int nt = omp_get_max_threads();
    if (nt <= 1) {
        for (int64_t i = 1; i < N; ++i)
            row_kernel(a + i * N, a + (i - 1) * N, M);
        return;
    }

    /* Pipeline block size: enough rows to amortize sync, small enough
     * to keep the pipeline full. */
    int64_t B = nrows / (4 * nt);
    if (B < 8) B = 8;
    int64_t nb = (nrows + B - 1) / B;

    flag_t *done = (flag_t *)calloc((size_t)nb, sizeof(flag_t));
    if (!done) return;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        for (int64_t k = 0; ; ++k) {
            int64_t b = tid + (int64_t)k * nthr;
            if (b >= nb) break;
            if (b > 0) {
                while (atomic_load_explicit(&done[b - 1].v, memory_order_acquire) == 0)
                    _mm_pause();
            }
            int64_t i0 = 1 + b * B;
            int64_t i1 = i0 + B;
            if (i1 > N) i1 = N;
            for (int64_t i = i0; i < i1; ++i)
                row_kernel(a + i * N, a + (i - 1) * N, M);
            atomic_store_explicit(&done[b].v, 1, memory_order_release);
        }
    }
    free(done);
}
