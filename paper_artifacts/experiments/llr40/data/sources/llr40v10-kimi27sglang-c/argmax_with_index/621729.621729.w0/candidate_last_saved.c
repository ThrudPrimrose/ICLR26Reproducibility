#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

typedef struct {
    double val;
    int64_t idx;
} maxloc_t;

static inline maxloc_t argmax_scalar(const double *a, int64_t start, int64_t end) {
    maxloc_t m = {a[start], start};
    for (int64_t i = start + 1; i < end; ++i) {
        if (__builtin_expect(a[i] > m.val, 0)) {
            m.val = a[i];
            m.idx = i;
        }
    }
    return m;
}

static inline maxloc_t argmax_avx512(const double *a, int64_t start, int64_t end) {
    int64_t n = end - start;
    if (n < 8) {
        return argmax_scalar(a, start, end);
    }

    const __m512i offsets = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i inc8 = _mm512_set1_epi64(8);
    __m512d vmax = _mm512_loadu_pd(a + start);
    __m512i vidx = _mm512_add_epi64(_mm512_set1_epi64(start), offsets);
    __m512i vwin = vidx;

    int64_t i = start + 8;
    for (; i + 8 <= end; i += 8) {
        vwin = _mm512_add_epi64(vwin, inc8);
        __m512d va = _mm512_loadu_pd(a + i);
        __mmask8 mask = _mm512_cmp_pd_mask(va, vmax, _CMP_GT_OQ);
        vmax = _mm512_mask_blend_pd(mask, vmax, va);
        vidx = _mm512_mask_blend_epi64(mask, vidx, vwin);
    }

    double vals[8];
    int64_t idxs[8];
    _mm512_storeu_pd(vals, vmax);
    _mm512_storeu_si512((__m512i *)idxs, vidx);

    maxloc_t m = {vals[0], idxs[0]};
    for (int k = 1; k < 8; ++k) {
        if (vals[k] > m.val || (vals[k] == m.val && idxs[k] < m.idx)) {
            m.val = vals[k];
            m.idx = idxs[k];
        }
    }

    for (; i < end; ++i) {
        if (__builtin_expect(a[i] > m.val, 0)) {
            m.val = a[i];
            m.idx = i;
        }
    }

    return m;
}

typedef struct {
    double val;
    int64_t idx;
    char pad[48];
} __attribute__((aligned(64))) maxloc_padded_t;

#define MAX_LOCALS 256

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        out_value[0] = 0.0;
        out_index[0] = -1;
        return;
    }

    if (__builtin_expect(isnan(a[0]), 0)) {
        out_value[0] = a[0];
        out_index[0] = 0;
        return;
    }

    if (LEN_1D < 256) {
        maxloc_t m = argmax_scalar(a, 0, LEN_1D);
        out_value[0] = m.val;
        out_index[0] = m.idx;
        return;
    }

    if (LEN_1D < 1048576) {
        maxloc_t m = argmax_avx512(a, 0, LEN_1D);
        out_value[0] = m.val;
        out_index[0] = m.idx;
        return;
    }

    int nthreads = omp_get_max_threads();
    maxloc_padded_t *locals;
    maxloc_padded_t stack_locals[MAX_LOCALS] __attribute__((aligned(64)));
    if (nthreads <= MAX_LOCALS) {
        locals = stack_locals;
    } else {
        locals = (maxloc_padded_t *)malloc(nthreads * sizeof(maxloc_padded_t));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t start = ((int64_t)tid * LEN_1D) / nthreads;
        int64_t end = ((int64_t)(tid + 1) * LEN_1D) / nthreads;
        if (start < end) {
            maxloc_t m = argmax_avx512(a, start, end);
            locals[tid].val = m.val;
            locals[tid].idx = m.idx;
        } else {
            locals[tid].val = -INFINITY;
            locals[tid].idx = INT64_MAX;
        }
    }

    maxloc_t m = {locals[0].val, locals[0].idx};
    for (int t = 1; t < nthreads; ++t) {
        if (locals[t].val > m.val || (locals[t].val == m.val && locals[t].idx < m.idx)) {
            m.val = locals[t].val;
            m.idx = locals[t].idx;
        }
    }

    if (nthreads > MAX_LOCALS) {
        free(locals);
    }

    out_value[0] = m.val;
    out_index[0] = m.idx;
}
