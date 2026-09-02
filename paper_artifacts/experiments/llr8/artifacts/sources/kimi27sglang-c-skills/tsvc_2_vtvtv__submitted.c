#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

/* Two-wide AVX-512 unroll of a[i] = a[i] * b[i] * c[i].
 * The loop is embarrassingly parallel and the arrays are accessed with unit stride,
 * so we thread the outer loop and vectorize 8 doubles per iteration. */

static inline void vtvtv_8(double *restrict a, const double *restrict b, const double *restrict c, int64_t i) {
    __m512d va = _mm512_loadu_pd(&a[i]);
    __m512d vb = _mm512_loadu_pd(&b[i]);
    __m512d vc = _mm512_loadu_pd(&c[i]);
    va = _mm512_mul_pd(va, vb);
    va = _mm512_mul_pd(va, vc);
    _mm512_storeu_pd(&a[i], va);
}

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    const int64_t main = n & ~15;   /* round down to a multiple of 16 doubles */

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < main; i += 16) {
        vtvtv_8(a, b, c, i);
        vtvtv_8(a, b, c, i + 8);
    }

    for (int64_t i = main; i < n; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }
}
