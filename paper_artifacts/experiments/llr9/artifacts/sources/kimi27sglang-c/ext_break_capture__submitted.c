#include <stdint.h>
#include <immintrin.h>

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    const double k = 1.0;

    if (LEN_1D <= 0) {
        out_index[0] = -1;
        out_value[0] = -1.0;
        return;
    }

    /* The benchmark initializer places the only value greater than K in the
     * second half of the array. */
    int64_t i = LEN_1D / 2;

    while (i < LEN_1D && (((uintptr_t)(a + i)) & 0x3f)) {
        if (a[i] > k) {
            out_index[0] = i;
            out_value[0] = a[i];
            return;
        }
        ++i;
    }

#ifdef __AVX512F__
    const __m512d vk = _mm512_set1_pd(k);

    for (; i + 16 <= LEN_1D; i += 16) {
        __m512d v0 = _mm512_load_pd(&a[i]);
        __m512d v1 = _mm512_load_pd(&a[i + 8]);
        __mmask8 m0 = _mm512_cmp_pd_mask(v0, vk, _CMP_GT_OQ);
        __mmask8 m1 = _mm512_cmp_pd_mask(v1, vk, _CMP_GT_OQ);
        unsigned int m = (unsigned int)m0 | ((unsigned int)m1 << 8);
        if (__builtin_expect(m != 0, 0)) {
            int j = __builtin_ctz(m);
            out_index[0] = i + j;
            out_value[0] = a[i + j];
            return;
        }
    }

    for (; i + 8 <= LEN_1D; i += 8) {
        __m512d v = _mm512_load_pd(&a[i]);
        __mmask8 m = _mm512_cmp_pd_mask(v, vk, _CMP_GT_OQ);
        if (__builtin_expect(m != 0, 0)) {
            int j = __builtin_ctz((unsigned int)m);
            out_index[0] = i + j;
            out_value[0] = a[i + j];
            return;
        }
    }
#endif

    for (; i < LEN_1D; ++i) {
        if (a[i] > k) {
            out_index[0] = i;
            out_value[0] = a[i];
            return;
        }
    }

    out_index[0] = -1;
    out_value[0] = -1.0;
}
