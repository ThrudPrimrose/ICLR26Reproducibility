#include <stdint.h>
#include <stddef.h>
#include <omp.h>
#include <immintrin.h>

static inline void run_block(double *a,
                             const double *restrict b,
                             const double *restrict c,
                             double *restrict d,
                             int64_t lo, int64_t hi,
                             double boundary)
{
    int64_t i = lo;

    if (hi - lo > 0) {
        uintptr_t addr = (uintptr_t)&a[i];
        int64_t peel = (int64_t)(((addr + 63ULL) & ~63ULL) - addr) / sizeof(double);
        if (peel > hi - lo) peel = hi - lo;
        for (; i < lo + peel; ++i) {
            double ci = c[i];
            double bi = b[i];
            double ai = bi + ci * ci + bi * bi + ci;
            double next = (i + 1 < hi) ? a[i + 1] : boundary;
            a[i] = ai;
            d[i] = ai + next;
        }
    }

    int64_t vec_limit = hi - 16;
    for (; i <= vec_limit; i += 8) {
        __m512d vc = _mm512_castsi512_pd(_mm512_stream_load_si512((void *)&c[i]));
        __m512d vb = _mm512_castsi512_pd(_mm512_stream_load_si512((void *)&b[i]));
        __m512d anext = _mm512_loadu_pd(&a[i + 1]);
        __m512d ai = _mm512_add_pd(_mm512_fmadd_pd(vc, vc, vc),
                                   _mm512_fmadd_pd(vb, vb, vb));
        _mm512_stream_pd(&a[i], ai);
        _mm512_stream_pd(&d[i], _mm512_add_pd(ai, anext));
    }

    for (; i < hi; ++i) {
        double ci = c[i];
        double bi = b[i];
        double ai = bi + ci * ci + bi * bi + ci;
        double next = (i + 1 < hi) ? a[i + 1] : boundary;
        a[i] = ai;
        d[i] = ai + next;
    }
}

void tsvc_2_s1244_fp64(double *restrict a, double *restrict b,
                       double *restrict c, double *restrict d,
                       int64_t LEN_1D, uint8_t *workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    int64_t n = LEN_1D - 1;
    if (n <= 0) return;

    if (n < 4096) {
        for (int64_t i = 0; i < n; ++i) {
            double ci = c[i];
            double bi = b[i];
            double ai = bi + ci * ci + bi * bi + ci;
            a[i] = ai;
            d[i] = ai + a[i + 1];
        }
        return;
    }

    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 256) nthreads = 256;

    int64_t start[256];
    int64_t end[256];
    double boundary[256];

    int64_t base = n / nthreads;
    int64_t rem = n % nthreads;
    int64_t s = 0;
    for (int t = 0; t < nthreads; ++t) {
        int64_t cnt = base + (t < rem ? 1 : 0);
        start[t] = s;
        s += cnt;
        end[t] = s;
        boundary[t] = (cnt > 0) ? a[end[t]] : 0.0;
    }

#pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int64_t lo = start[tid];
        int64_t hi = end[tid];
        if (lo < hi)
            run_block(a, b, c, d, lo, hi, boundary[tid]);
    }

    _mm_sfence();
}
