#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, double *restrict c,
                      double *restrict d, double *restrict e,
                      int64_t LEN_1D, uint8_t *workspace, int64_t workspace_size)
{
    const int64_t n = LEN_1D;
    const int64_t nvec = n & ~15;

    #pragma omp parallel for schedule(static, 2048)
    for (int64_t i = 0; i < nvec; i += 16) {
        __m512d vd0 = _mm512_loadu_pd(&d[i]);
        __m512d ve0 = _mm512_loadu_pd(&e[i]);
        __m512d vb0 = _mm512_mul_pd(vd0, ve0);
        __m512d va0 = _mm512_loadu_pd(&a[i]);
        __m512d vc0 = _mm512_loadu_pd(&c[i]);
        va0 = _mm512_fmadd_pd(vc0, vb0, va0);
        _mm512_storeu_pd(&a[i], va0);
        _mm512_stream_pd(&b[i], vb0);

        __m512d vd1 = _mm512_loadu_pd(&d[i+8]);
        __m512d ve1 = _mm512_loadu_pd(&e[i+8]);
        __m512d vb1 = _mm512_mul_pd(vd1, ve1);
        __m512d va1 = _mm512_loadu_pd(&a[i+8]);
        __m512d vc1 = _mm512_loadu_pd(&c[i+8]);
        va1 = _mm512_fmadd_pd(vc1, vb1, va1);
        _mm512_storeu_pd(&a[i+8], va1);
        _mm512_stream_pd(&b[i+8], vb1);
    }

    _mm_sfence();

    for (int64_t i = nvec; i < n; ++i) {
        double bi = d[i] * e[i];
        b[i] = bi;
        a[i] = a[i] + bi * c[i];
    }
}
