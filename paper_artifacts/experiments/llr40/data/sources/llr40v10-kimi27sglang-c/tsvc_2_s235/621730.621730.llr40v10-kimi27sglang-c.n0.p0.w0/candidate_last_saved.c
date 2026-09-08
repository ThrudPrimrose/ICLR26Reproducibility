#include <stdint.h>
#include <omp.h>

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;

    a = __builtin_assume_aligned(a, 64);
    aa = __builtin_assume_aligned(aa, 64);
    b = __builtin_assume_aligned(b, 64);
    bb = __builtin_assume_aligned(bb, 64);
    c = __builtin_assume_aligned(c, 64);

    if (unlikely(n < 256)) {
#pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            a[i] += b[i] * c[i];
        }
        for (int64_t j = 1; j < n; ++j) {
            const int64_t row = j * n;
            const int64_t prev = row - n;
#pragma omp simd
            for (int64_t i = 0; i < n; ++i) {
                aa[row + i] = aa[prev + i] + bb[row + i] * a[i];
            }
        }
        return;
    }

#pragma omp parallel num_threads(24)
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t chunk = (n + nthreads - 1) / nthreads;
        const int64_t chunk_a = (chunk + 7) & ~((int64_t)7);
        const int64_t i0 = tid * chunk_a;
        const int64_t i1 = (i0 + chunk_a < n) ? i0 + chunk_a : n;

        if (i0 < n) {
#pragma omp simd
            for (int64_t i = i0; i < i1; ++i) {
                a[i] += b[i] * c[i];
            }

            for (int64_t j = 1; j < n; ++j) {
                const int64_t row = j * n;
                const int64_t prev = row - n;
#pragma omp simd
                for (int64_t i = i0; i < i1; ++i) {
                    aa[row + i] = aa[prev + i] + bb[row + i] * a[i];
                }
            }
        }
    }
}
