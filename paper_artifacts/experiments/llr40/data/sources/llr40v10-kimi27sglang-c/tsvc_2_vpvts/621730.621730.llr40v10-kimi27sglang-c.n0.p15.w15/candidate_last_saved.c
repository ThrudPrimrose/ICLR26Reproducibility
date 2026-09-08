#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
    const double s = (double)S;
    const __m512d vs = _mm512_set1_pd(s);
    const int64_t n = LEN_1D & ~7;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i += 8) {
        _mm_prefetch((const char *)&a[i + 128], _MM_HINT_T0);
        _mm_prefetch((const char *)&b[i + 128], _MM_HINT_T0);
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        va = _mm512_fmadd_pd(vb, vs, va);
        _mm512_storeu_pd(&a[i], va);
    }

    for (int64_t i = n; i < LEN_1D; ++i) {
        a[i] += b[i] * s;
    }
}
