#include <stdint.h>
#include <stdatomic.h>
#include <omp.h>

#ifndef NT_MAX
#define NT_MAX 128
#endif
#ifndef SERIAL_THRESHOLD
#define SERIAL_THRESHOLD 512
#endif

static inline void pause_cpu(void)
{
    __builtin_ia32_pause();
}

typedef struct { _Atomic int v; char pad[64 - sizeof(_Atomic int)]; } padded_atomic;

void wf_diff_skew_fp64(double *restrict a, int64_t LEN_2D,
                       uint8_t *restrict workspace, int64_t workspace_bytes)
{
    const int64_t n = LEN_2D;
    const int64_t nj = n - 1;

    if (n < SERIAL_THRESHOLD || omp_get_max_threads() <= 1) {
        for (int64_t i = 1; i < n; ++i) {
            for (int64_t j = 0; j < nj; ++j) {
                const int64_t idx = i * n + j;
                a[idx] = a[idx] + a[idx - n] + a[idx - n + 1];
            }
        }
        return;
    }

    padded_atomic ready[NT_MAX];

    #pragma omp parallel default(none) shared(a, n, nj, ready)
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        if (tid == 0) {
            for (int t = 0; t < nt; ++t) atomic_store_explicit(&ready[t].v, -1, memory_order_relaxed);
            atomic_store_explicit(&ready[0].v, 0, memory_order_release);
        }
        #pragma omp barrier

        // thread 0 = rightmost column chunk, thread nt-1 = leftmost
        const int64_t base = nj / nt;
        const int64_t rem = nj % nt;
        int64_t c0, c1;
        if (tid < (int)rem) {
            c1 = nj - tid * (base + 1);
            c0 = c1 - (base + 1);
        } else {
            c1 = nj - rem * (base + 1) - (tid - rem) * base;
            c0 = c1 - base;
        }
        if (c0 < 0) c0 = 0;

        for (int64_t i = 1; i < n; ++i) {
            if (c1 - 1 > c0) {
                #pragma omp simd
                for (int64_t j = c0; j < c1 - 1; ++j) {
                    const int64_t idx = i * n + j;
                    a[idx] = a[idx] + a[idx - n] + a[idx - n + 1];
                }
            }
            if (tid > 0) {
                while (atomic_load_explicit(&ready[tid - 1].v, memory_order_acquire) < i - 1) {
                    pause_cpu();
                }
            }
            if (c1 > c0) {
                const int64_t j = c1 - 1;
                const int64_t idx = i * n + j;
                a[idx] = a[idx] + a[idx - n] + a[idx - n + 1];
            }
            atomic_store_explicit(&ready[tid].v, (int)i, memory_order_release);
        }
    }
}
