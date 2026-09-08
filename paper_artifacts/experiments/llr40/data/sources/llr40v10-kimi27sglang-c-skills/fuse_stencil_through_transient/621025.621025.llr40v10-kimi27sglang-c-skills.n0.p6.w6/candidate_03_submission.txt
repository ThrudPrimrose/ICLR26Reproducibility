#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    const int64_t total = n - 3;  // indices 1 .. n-3 inclusive

    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t start = 1 + (tid * total) / nt;
        const int64_t end = 1 + ((tid + 1) * total) / nt;

        int64_t i = start;

        // scalar peel to align out+i to 64 bytes
        while (i < end && ((uintptr_t)(out + i) & 63)) {
            double t0 = a[i - 1] + a[i] + a[i + 1];
            double t1 = a[i] + a[i + 1] + a[i + 2];
            out[i] = t0 * t1;
            ++i;
        }

        const int64_t vec_end = end - 7;
        for (; i < vec_end; i += 8) {
            __builtin_prefetch((const void *)(a + i + 256), 0, 3);
            __m512d v0 = _mm512_loadu_pd(a + i - 1);
            __m512d v1 = _mm512_loadu_pd(a + i);
            __m512d v2 = _mm512_loadu_pd(a + i + 1);
            __m512d v3 = _mm512_loadu_pd(a + i + 2);
            __m512d t0 = _mm512_add_pd(_mm512_add_pd(v0, v1), v2);
            __m512d t1 = _mm512_add_pd(_mm512_add_pd(v1, v2), v3);
            _mm512_store_pd(out + i, _mm512_mul_pd(t0, t1));
        }

        for (; i < end; ++i) {
            double t0 = a[i - 1] + a[i] + a[i + 1];
            double t1 = a[i] + a[i + 1] + a[i + 2];
            out[i] = t0 * t1;
        }
    }
}
