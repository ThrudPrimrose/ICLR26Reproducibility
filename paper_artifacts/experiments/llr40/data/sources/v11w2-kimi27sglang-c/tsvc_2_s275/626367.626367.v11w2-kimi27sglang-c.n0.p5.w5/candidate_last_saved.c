#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s275_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    if (n <= 0) return;

    const int64_t block = 8;
    const int64_t nblocks = (n + block - 1) / block;

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblocks; b++) {
        const int64_t i0 = b * block;
        const int64_t i1 = (i0 + block < n) ? i0 + block : n;
        const int64_t len = i1 - i0;

        if (len == block) {
            __m512d prev = _mm512_loadu_pd(&aa[i0]);
            const __mmask8 mask = _mm512_cmp_pd_mask(prev, _mm512_setzero_pd(), _CMP_GT_OQ);
            if (mask != 0) {
                for (int64_t j = 1; j < n; j++) {
                    const int64_t row = j * n + i0;
                    const int64_t next_row = row + n;
                    _mm_prefetch((const char *)&bb[next_row], _MM_HINT_T1);
                    _mm_prefetch((const char *)&cc[next_row], _MM_HINT_T1);
                    const __m512d b_col = _mm512_loadu_pd(&bb[row]);
                    const __m512d c_col = _mm512_loadu_pd(&cc[row]);
                    prev = _mm512_fmadd_pd(b_col, c_col, prev);
                    _mm512_mask_storeu_pd(&aa[row], mask, prev);
                }
            }
        } else {
            for (int64_t i = i0; i < i1; i++) {
                if (aa[i] > 0.0) {
                    for (int64_t j = 1; j < n; j++) {
                        const int64_t idx = j * n + i;
                        aa[idx] = aa[idx - n] + bb[idx] * cc[idx];
                    }
                }
            }
        }
    }
}
