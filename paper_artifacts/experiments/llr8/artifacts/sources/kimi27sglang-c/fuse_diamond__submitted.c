#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline void scalar_loop(const double *restrict a, double *restrict out, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        double t = a[i] * a[i];
        out[i] = (t + 1.0) * (t - 1.0);
    }
}

static inline void simd_loop(const double *restrict a, double *restrict out, int64_t n)
{
    int64_t i = 0;
    __m512d one = _mm512_set1_pd(1.0);
    int64_t limit = n & ~7;
    for (; i < limit; i += 8) {
        __m512d ai = _mm512_loadu_pd(a + i);
        __m512d t = _mm512_mul_pd(ai, ai);
        __m512d tp = _mm512_add_pd(t, one);
        __m512d tm = _mm512_sub_pd(t, one);
        __m512d r = _mm512_mul_pd(tp, tm);
        _mm512_storeu_pd(out + i, r);
    }
    scalar_loop(a + i, out + i, n - i);
}

static __attribute__((noinline)) void compute_omp(const double *restrict a, double *restrict out, int64_t n)
{
    int64_t limit = n & ~7;
#pragma omp parallel for schedule(static) default(none) shared(a, out, n, limit)
    for (int64_t i = 0; i < limit; i += 8) {
        __m512d one = _mm512_set1_pd(1.0);
        __m512d ai = _mm512_loadu_pd(a + i);
        __m512d t = _mm512_mul_pd(ai, ai);
        __m512d tp = _mm512_add_pd(t, one);
        __m512d tm = _mm512_sub_pd(t, one);
        __m512d r = _mm512_mul_pd(tp, tm);
        _mm512_storeu_pd(out + i, r);
    }
    scalar_loop(a + limit, out + limit, n - limit);
}

static __attribute__((noinline)) void compute_omp_nt(const double *restrict a, double *restrict out, int64_t n)
{
    int64_t limit = n & ~7;
#pragma omp parallel for schedule(static) default(none) shared(a, out, n, limit)
    for (int64_t i = 0; i < limit; i += 8) {
        __m512d one = _mm512_set1_pd(1.0);
        __m512d ai = _mm512_loadu_pd(a + i);
        __m512d t = _mm512_mul_pd(ai, ai);
        __m512d tp = _mm512_add_pd(t, one);
        __m512d tm = _mm512_sub_pd(t, one);
        __m512d r = _mm512_mul_pd(tp, tm);
        _mm512_stream_pd(out + i, r);
    }
    _mm_sfence();
    scalar_loop(a + limit, out + limit, n - limit);
}

void fuse_diamond_fp64(const double *restrict a, double *restrict out, int64_t LEN_1D,
                       uint8_t *restrict workspace, int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    if (LEN_1D <= 0)
        return;

    if (LEN_1D > 65536) {
        compute_omp_nt(a, out, LEN_1D);
    } else if (LEN_1D > 8192) {
        compute_omp(a, out, LEN_1D);
    } else {
        simd_loop(a, out, LEN_1D);
    }
}
