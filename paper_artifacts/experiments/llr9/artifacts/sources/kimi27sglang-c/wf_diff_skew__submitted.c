#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

static inline void serial_row(double *restrict cur, const double *restrict prev, int64_t n) {
    const int64_t vlen = (n - 1) & ~7;
    int64_t j = 0;
    for (; j < vlen; j += 8) {
        __m512d c0 = _mm512_loadu_pd(cur + j);
        __m512d p0 = _mm512_loadu_pd(prev + j);
        __m512d p1 = _mm512_loadu_pd(prev + j + 1);
        c0 = _mm512_add_pd(c0, p0);
        c0 = _mm512_add_pd(c0, p1);
        _mm512_storeu_pd(cur + j, c0);
    }
    for (; j < n - 1; ++j) {
        cur[j] = cur[j] + prev[j] + prev[j + 1];
    }
}

void wf_diff_skew_fp64(double *restrict a, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    if (N <= 1) return;

    if (N < 512) {
        for (int64_t i = 1; i < N; ++i) {
            serial_row(a + i * N, a + (i - 1) * N, N);
        }
        return;
    }

    const int64_t M = N - 1;
    const int64_t nchunks = (M + 7) / 8;

    int max_threads = omp_get_max_threads();
    if (max_threads < 1) max_threads = 1;
    int nt = (nchunks < max_threads) ? (int)nchunks : max_threads;
    if (nt < 1) nt = 1;

    static int64_t *ready = NULL;
    static int ready_cap = 0;
    if (nt > ready_cap) {
        int64_t *new_ready = (int64_t *)aligned_alloc(64, (size_t)nt * 64);
        if (!new_ready) return;
        free(ready);
        ready = new_ready;
        ready_cap = nt;
    }
    for (int k = 0; k < nt; ++k) ready[k * 8] = 0;

    #pragma omp parallel num_threads(nt)
    {
        const int p = omp_get_thread_num();
        const int64_t chunk0 = (int64_t)p * nchunks / nt;
        const int64_t chunk1 = ((int64_t)p + 1) * nchunks / nt;
        const int64_t first = chunk0 * 8;
        const int64_t last = (p == nt - 1) ? (M - 1) : (chunk1 * 8 - 1);
        const int has_right = (p != nt - 1);

        if (first > last) {
            __atomic_store_n(&ready[p * 8], (int64_t)N, __ATOMIC_RELEASE);
        } else {
            for (int64_t i = 1; i < N; ++i) {
                const double *restrict prev = a + (i - 1) * N;
                double *restrict cur = a + i * N;

                if (first == last) {
                    if (has_right) {
                        while (__atomic_load_n(&ready[(p + 1) * 8], __ATOMIC_ACQUIRE) < i - 1) {
                            _mm_pause();
                        }
                    }
                    cur[first] = cur[first] + prev[first] + prev[first + 1];
                } else {
                    cur[first] = cur[first] + prev[first] + prev[first + 1];
                    __atomic_store_n(&ready[p * 8], i, __ATOMIC_RELEASE);

                    int64_t j = first + 1;
                    for (; j + 7 <= last - 1; j += 8) {
                        __m512d c0 = _mm512_loadu_pd(cur + j);
                        __m512d p0 = _mm512_loadu_pd(prev + j);
                        __m512d p1 = _mm512_loadu_pd(prev + j + 1);
                        c0 = _mm512_add_pd(c0, p0);
                        c0 = _mm512_add_pd(c0, p1);
                        _mm512_storeu_pd(cur + j, c0);
                    }
                    for (; j <= last - 1; ++j) {
                        cur[j] = cur[j] + prev[j] + prev[j + 1];
                    }

                    if (has_right) {
                        while (__atomic_load_n(&ready[(p + 1) * 8], __ATOMIC_ACQUIRE) < i - 1) {
                            _mm_pause();
                        }
                    }
                    cur[last] = cur[last] + prev[last] + prev[last + 1];
                }
                __atomic_store_n(&ready[p * 8], i, __ATOMIC_RELEASE);
            }
        }
    }
}
