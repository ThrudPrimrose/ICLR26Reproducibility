#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 const double *restrict src,
                                 const double *restrict weight,
                                 const int64_t LEN_1D) {
    int64_t n = 0;
    const __m512d zero = _mm512_setzero_pd();

    int64_t i = 0;
    for (; i + 8 <= LEN_1D; i += 8) {
        const __m512d s = _mm512_loadu_pd(src + i);
        const __mmask8 m = _mm512_cmp_pd_mask(s, zero, _CMP_GT_OQ);
        if (m) {
            const __m512d w = _mm512_loadu_pd(weight + i);
            const __m512d v = _mm512_mul_pd(s, w);
            _mm512_mask_compressstoreu_pd(packed + n, m, v);
            n += (int64_t)_mm_popcnt_u32((unsigned)m);
        }
    }
    for (; i < LEN_1D; ++i) {
        const double s = src[i];
        if (s > 0.0) {
            packed[n++] = s * weight[i];
        }
    }

    out_count[0] = n;
}
