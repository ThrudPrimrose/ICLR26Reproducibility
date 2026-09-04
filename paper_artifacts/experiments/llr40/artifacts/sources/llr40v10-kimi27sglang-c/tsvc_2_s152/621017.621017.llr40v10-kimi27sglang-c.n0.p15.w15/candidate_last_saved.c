#include <stdint.h>
#include <immintrin.h>

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    const int64_t nvec = n & ~7;
#pragma omp parallel for
    for (int64_t i = 0; i < nvec; i += 8) {
        __m512d dd = _mm512_loadu_pd(d + i);
        __m512d ee = _mm512_loadu_pd(e + i);
        __m512d cc = _mm512_loadu_pd(c + i);
        __m512d aa = _mm512_loadu_pd(a + i);
        __m512d bb = _mm512_mul_pd(dd, ee);
        aa = _mm512_fmadd_pd(bb, cc, aa);
        _mm512_storeu_pd(b + i, bb);
        _mm512_storeu_pd(a + i, aa);
    }
#pragma omp parallel for
    for (int64_t i = nvec; i < n; ++i) {
        double bi = d[i] * e[i];
        b[i] = bi;
        a[i] += bi * c[i];
    }
}
