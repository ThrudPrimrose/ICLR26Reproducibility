#include <stdint.h>
#include <immintrin.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    int64_t i = 0;
    const int64_t N = LEN_1D;
    
    // Process 8 doubles per iteration with AVX-512
    for (; i + 8 <= N; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vc = _mm512_loadu_pd(&c[i]);
        __m512d vr = _mm512_mul_pd(va, _mm512_mul_pd(vb, vc));
        _mm512_storeu_pd(&a[i], vr);
    }
    
    // Scalar tail
    for (; i < N; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }
}
