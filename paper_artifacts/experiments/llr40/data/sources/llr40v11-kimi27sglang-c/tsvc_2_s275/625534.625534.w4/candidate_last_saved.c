#include <stdint.h>
#include <immintrin.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb,
                      const double *restrict cc, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    if (n <= 1) return;

    const int64_t chunks = n & ~((int64_t)7);

    #pragma omp parallel for schedule(dynamic, 8)
    for (int64_t i = 0; i < chunks; i += 8) {
        __m512d prev = _mm512_loadu_pd(&aa[i]);
        __mmask8 mask = _mm512_cmp_pd_mask(prev, _mm512_setzero_pd(), _CMP_GT_OQ);
        if (mask == 0) continue;

        for (int64_t j = 1; j < n; ++j) {
            const int64_t row_off = j * n;
            __m512d b = _mm512_loadu_pd(&bb[row_off + i]);
            __m512d c = _mm512_loadu_pd(&cc[row_off + i]);
            prev = _mm512_fmadd_pd(b, c, prev);
            _mm512_mask_storeu_pd(&aa[row_off + i], mask, prev);
        }
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = chunks; i < n; ++i) {
        if (!(aa[i] > 0.0)) continue;
        for (int64_t j = 1; j < n; ++j) {
            aa[j * n + i] = aa[(j - 1) * n + i] + bb[j * n + i] * cc[j * n + i];
        }
    }
}
