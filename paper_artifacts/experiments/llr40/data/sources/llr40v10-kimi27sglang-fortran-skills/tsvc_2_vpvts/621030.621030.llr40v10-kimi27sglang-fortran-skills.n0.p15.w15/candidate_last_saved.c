#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t n, const int64_t s) {
    const double scale = (double)s;
    const int64_t n8 = n & ~7;
    const __m512d vs = _mm512_set1_pd(scale);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n8; i += 8) {
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vr = _mm512_fmadd_pd(vb, vs, va);
        _mm512_stream_pd(&a[i], vr);
    }
    _mm_sfence();
    for (int64_t i = n8; i < n; ++i) {
        a[i] += b[i] * scale;
    }
}
