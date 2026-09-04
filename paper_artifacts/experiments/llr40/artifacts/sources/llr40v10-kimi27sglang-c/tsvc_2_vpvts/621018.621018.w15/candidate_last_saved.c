#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b, const int64_t LEN_1D, const int64_t S) {
    const double s = (double)S;
    const __m512d vs = _mm512_set1_pd(s);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < LEN_1D; i += 8) {
        int64_t rem = LEN_1D - i;
        if (rem >= 8) {
            __m512d vb = _mm512_loadu_pd(&b[i]);
            __m512d va = _mm512_loadu_pd(&a[i]);
            __m512d vr = _mm512_fmadd_pd(vb, vs, va);
            _mm512_storeu_pd(&a[i], vr);
        } else {
            __mmask8 mask = (__mmask8)((1U << rem) - 1);
            __m512d vb = _mm512_maskz_loadu_pd(mask, &b[i]);
            __m512d va = _mm512_maskz_loadu_pd(mask, &a[i]);
            __m512d vr = _mm512_fmadd_pd(vb, vs, va);
            _mm512_mask_storeu_pd(&a[i], mask, vr);
        }
    }
}
