#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <omp.h>
#include <x86intrin.h>

void tsvc_2_s119_fp64(
    double *restrict aa,
    const double *restrict bb,
    const int64_t LEN_2D,
    uint8_t *restrict workspace,
    const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    const int64_t N = LEN_2D;
    if (N < 2) return;
    const int64_t ncol = N - 1;

    if (N < 128) {
        for (int64_t i = 1; i < N; ++i) {
            double *restrict a = aa + i * N;
            const double *restrict ap = aa + (i - 1) * N;
            const double *restrict b = bb + i * N;
            for (int64_t j = 1; j < N; ++j)
                a[j] = ap[j - 1] + b[j];
        }
        return;
    }

    int T = omp_get_max_threads();
    if (T < 1) T = 1;
    if (T > (int)ncol) T = (int)ncol;

    static _Alignas(64) _Atomic int32_t cnt[1024];
    for (int t = 0; t < T; ++t)
        atomic_store_explicit(&cnt[t], 0, memory_order_relaxed);

    #pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const int64_t base = ncol / T;
        const int64_t rem = ncol % T;
        int64_t c0 = 1;
        for (int q = 0; q < t; ++q) c0 += base + (q < rem);
        const int64_t m = base + (t < rem);

        for (int64_t r = 1; r < N; ++r) {
            if (t > 0)
                while (atomic_load_explicit(&cnt[t - 1], memory_order_acquire) < r - 1)
                    _mm_pause();
            if (r + 1 < N) {
                const char *pn = (const char *)(bb + (r + 1) * N + c0);
                for (int64_t j = 0; j < m; j += 8)
                    _mm_prefetch(pn + j * 8, _MM_HINT_T0);
            }
            double *restrict a = (double *restrict)(aa + r * N + c0);
            const double *restrict ap = (const double *restrict)(aa + (r - 1) * N + c0 - 1);
            const double *restrict b = (const double *restrict)(bb + r * N + c0);
            for (int64_t j = 0; j < m; ++j)
                a[j] = ap[j] + b[j];
            atomic_store_explicit(&cnt[t], (int32_t)r, memory_order_release);
        }
    }
}
