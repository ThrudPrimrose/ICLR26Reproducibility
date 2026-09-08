#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline void stencil_serial(const double *restrict a, double *restrict out, int64_t i0, int64_t i1) {
    int64_t i = i0;
    for (; i + 8 <= i1; i += 8) {
        __m512d v0 = _mm512_loadu_pd(&a[i - 1]);
        __m512d v1 = _mm512_loadu_pd(&a[i]);
        __m512d v2 = _mm512_loadu_pd(&a[i + 1]);
        __m512d v3 = _mm512_loadu_pd(&a[i + 2]);
        __m512d s12  = _mm512_add_pd(v1, v2);
        __m512d left = _mm512_add_pd(s12, v0);
        __m512d right= _mm512_add_pd(s12, v3);
        _mm512_storeu_pd(&out[i], _mm512_mul_pd(left, right));
    }
    for (; i < i1; ++i) {
        out[i] = (a[i - 1] + a[i] + a[i + 1]) * (a[i] + a[i + 1] + a[i + 2]);
    }
}

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    int64_t start = 1;
    int64_t end = LEN_1D - 2;
    if (end <= start) return;
    int64_t count = end - start;
    if (count < 200000) {
        stencil_serial(a, out, start, end);
        return;
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        // Round chunk up to a multiple of 8 elements (64 bytes) to avoid false sharing.
        int64_t base = (count + nthreads - 1) / nthreads;
        int64_t chunk = (base + 7) & ~((int64_t)7);
        int64_t i0 = start + (int64_t)tid * chunk;
        int64_t i1 = i0 + chunk;
        if (i1 > end) i1 = end;
        if (i0 < end) {
            stencil_serial(a, out, i0, i1);
        }
    }
}
