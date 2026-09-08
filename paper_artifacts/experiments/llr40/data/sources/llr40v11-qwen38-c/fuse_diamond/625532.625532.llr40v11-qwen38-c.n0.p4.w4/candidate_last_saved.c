#include <stdint.h>
#include <omp.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX512F__
static void block(const double *a, double *out, int64_t n) {
    int64_t i = 0;
    const int64_t n32 = n & ~31LL;
    const __m512d one = _mm512_set1_pd(1.0);
    for (; i < n32; i += 32) {
        __m512d x0 = _mm512_loadu_pd(a + i);
        __m512d x1 = _mm512_loadu_pd(a + i + 8);
        __m512d x2 = _mm512_loadu_pd(a + i + 16);
        __m512d x3 = _mm512_loadu_pd(a + i + 24);
        __m512d t0 = _mm512_mul_pd(x0, x0);
        __m512d t1 = _mm512_mul_pd(x1, x1);
        __m512d t2 = _mm512_mul_pd(x2, x2);
        __m512d t3 = _mm512_mul_pd(x3, x3);
        _mm512_storeu_pd(out + i,      _mm512_mul_pd(_mm512_add_pd(t0, one), _mm512_sub_pd(t0, one)));
        _mm512_storeu_pd(out + i + 8,  _mm512_mul_pd(_mm512_add_pd(t1, one), _mm512_sub_pd(t1, one)));
        _mm512_storeu_pd(out + i + 16, _mm512_mul_pd(_mm512_add_pd(t2, one), _mm512_sub_pd(t2, one)));
        _mm512_storeu_pd(out + i + 24, _mm512_mul_pd(_mm512_add_pd(t3, one), _mm512_sub_pd(t3, one)));
    }
    for (; i < n; ++i) {
        double t = a[i] * a[i];
        out[i] = (t + 1.0) * (t - 1.0);
    }
}
#else
static void block(const double *a, double *out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        double t = a[i] * a[i];
        out[i] = (t + 1.0) * (t - 1.0);
    }
}
#endif

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    if (LEN_1D < 131072) { block(a, out, LEN_1D); return; }
    const int64_t chunk = 65536;
    #pragma omp parallel
    {
        #pragma omp for schedule(static, 1)
        for (int64_t base = 0; base < LEN_1D; base += chunk) {
            int64_t n = chunk < LEN_1D - base ? chunk : LEN_1D - base;
            block(a + base, out + base, n);
        }
    }
}
