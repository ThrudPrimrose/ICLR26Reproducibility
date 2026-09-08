#include <stdint.h>
#include <immintrin.h>

typedef struct {
    double value;
    int64_t index;
} argmax_pair_t;

static inline argmax_pair_t pair_max(argmax_pair_t a, argmax_pair_t b) {
    return (a.value > b.value) ? a : b;
}

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    argmax_pair_t best = {a[0], 0};

    int64_t i = 1;

    // AVX-512 main loop: 8 doubles per iteration.
    if (LEN_1D >= 9) {
        __m512d vbest_val = _mm512_set1_pd(a[0]);
        __m512i vbest_idx = _mm512_set1_epi64(0);
        __m512i offset = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);

        for (; i + 8 <= LEN_1D; i += 8) {
            __m512d v = _mm512_loadu_pd(&a[i]);
            __mmask8 mask = _mm512_cmp_pd_mask(v, vbest_val, _CMP_GT_OQ);
            vbest_val = _mm512_mask_blend_pd(mask, vbest_val, v);
            __m512i cur_idx = _mm512_add_epi64(_mm512_set1_epi64(i), offset);
            vbest_idx = _mm512_mask_blend_epi64(mask, vbest_idx, cur_idx);
        }

        alignas(64) double vals[8];
        alignas(64) int64_t idxs[8];
        _mm512_store_pd(vals, vbest_val);
        _mm512_store_si512((__m512i *)idxs, vbest_idx);

        for (int k = 0; k < 8; ++k) {
            argmax_pair_t cand = {vals[k], idxs[k]};
            best = pair_max(best, cand);
        }
    }

    // Scalar tail.
    for (; i < LEN_1D; ++i) {
        double v = a[i];
        if (v > best.value) {
            best.value = v;
            best.index = i;
        }
    }

    out_value[0] = best.value;
    out_index[0] = best.index;
}
