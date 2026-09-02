#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

typedef struct {
    double value;
    int64_t index;
} maxpair_t;

static inline maxpair_t maxloc_scalar(const double *restrict a, int64_t start, int64_t end) {
    maxpair_t best = { a[start], start };
    for (int64_t i = start + 1; i < end; i++) {
        double v = a[i];
        if (v > best.value) {
            best.value = v;
            best.index = i;
        }
    }
    return best;
}

static inline maxpair_t maxloc_avx512(const double *restrict a, int64_t start, int64_t end) {
    int64_t len = end - start;
    if (len < 16) {
        return maxloc_scalar(a, start, end);
    }

    __m512d best_val = _mm512_set1_pd(a[start]);
    __m512i best_idx = _mm512_set1_epi64(start);
    __m512i idx = _mm512_set_epi64(start + 7, start + 6, start + 5, start + 4,
                                    start + 3, start + 2, start + 1, start);
    __m512i idx_inc = _mm512_set1_epi64(8);

    int64_t i = start;
    int64_t last_full = start + ((len >> 3) << 3);
    for (; i < last_full; i += 8) {
        __m512d v = _mm512_loadu_pd(&a[i]);
        __mmask8 gt = _mm512_cmp_pd_mask(v, best_val, _CMP_GT_OQ);
        best_val = _mm512_mask_max_pd(best_val, gt, best_val, v);
        best_idx = _mm512_mask_blend_epi64(gt, best_idx, idx);
        idx = _mm512_add_epi64(idx, idx_inc);
    }

    double vals[8];
    int64_t idxs[8];
    _mm512_storeu_pd(vals, best_val);
    _mm512_storeu_si512(idxs, best_idx);

    maxpair_t best = { vals[0], idxs[0] };
    for (int k = 1; k < 8; k++) {
        double v = vals[k];
        if (v > best.value || (v == best.value && idxs[k] < best.index)) {
            best.value = v;
            best.index = idxs[k];
        }
    }

    for (; i < end; i++) {
        double v = a[i];
        if (v > best.value) {
            best.value = v;
            best.index = i;
        }
    }
    return best;
}

void argmax_with_index_fp64(double *restrict a, int64_t *restrict out_index, double *restrict out_value, int64_t LEN_1D, uint8_t *restrict workspace, int64_t workspace_size) {
    maxpair_t best = { a[0], 0 };

    if (LEN_1D <= 1) {
        *out_value = best.value;
        *out_index = best.index;
        return;
    }

    int nthreads = omp_get_max_threads();
    maxpair_t *partials = (maxpair_t *)aligned_alloc(64, (size_t)nthreads * sizeof(maxpair_t));

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int64_t total = LEN_1D - 1;
        int64_t start = 1 + (int64_t)tid * total / nthreads;
        int64_t end = 1 + (int64_t)(tid + 1) * total / nthreads;
        partials[tid] = maxloc_avx512(a, start, end);
    }

    for (int t = 0; t < nthreads; t++) {
        double v = partials[t].value;
        int64_t idx = partials[t].index;
        if (v > best.value || (v == best.value && idx < best.index)) {
            best.value = v;
            best.index = idx;
        }
    }

    free(partials);
    *out_value = best.value;
    *out_index = best.index;
}
