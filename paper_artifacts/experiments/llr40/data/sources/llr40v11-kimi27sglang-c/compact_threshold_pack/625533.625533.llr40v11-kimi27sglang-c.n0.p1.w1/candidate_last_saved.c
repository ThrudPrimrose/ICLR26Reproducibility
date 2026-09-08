#include <stdint.h>
#include <immintrin.h>

void compact_threshold_pack_fp64(int64_t *restrict out_count, double *restrict packed,
                                 double *restrict src, double *restrict weight,
                                 int64_t LEN_1D, uint8_t *restrict workspace,
                                 int64_t workspace_size) {
    (void)workspace;
    (void)workspace_size;

    const __m512d zero = _mm512_setzero_pd();
    double *out = packed;
    int64_t i = 0;

    for (; i + 8 <= LEN_1D; i += 8) {
        __m512d s = _mm512_loadu_pd(src + i);
        __mmask8 m = _mm512_cmp_pd_mask(s, zero, _CMP_GT_OQ);
        __m512d w = _mm512_loadu_pd(weight + i);
        __m512d p = _mm512_mul_pd(s, w);
        _mm512_mask_compressstoreu_pd(out, m, p);
        out += _mm_popcnt_u32(m);
    }

    for (; i < LEN_1D; ++i) {
        if (src[i] > 0.0) {
            *out++ = src[i] * weight[i];
        }
    }

    *out_count = out - packed;
}
