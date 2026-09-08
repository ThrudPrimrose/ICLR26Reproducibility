#include <stdint.h>
#include <float.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

typedef struct {
    double maxv;
    int64_t idx;
} maxloc_t;

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    if (LEN_2D <= 0) {
        bb[0] = 0.0;
        return;
    }

    const int64_t n = LEN_2D;
    const int maxt = omp_get_max_threads();
    maxloc_t *restrict partials = (maxloc_t *)malloc((size_t)maxt * sizeof(maxloc_t));
    for (int t = 0; t < maxt; ++t) {
        partials[t].maxv = -DBL_MAX;
        partials[t].idx = INT64_MAX;
    }
    partials[0].maxv = aa[0];
    partials[0].idx = 0;

    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t row0 = (n * tid) / nt;
        const int64_t row1 = (n * (tid + 1)) / nt;

        const __m512d negmax = _mm512_set1_pd(-DBL_MAX);
        const __m512i badidx = _mm512_set1_epi64(INT64_MAX);
        const __m512i inc8 = _mm512_set1_epi64(8);
        __m512d vmax0 = negmax;
        __m512d vmax1 = negmax;
        __m512i vidx0 = badidx;
        __m512i vidx1 = badidx;

        for (int64_t i = row0; i < row1; ++i) {
            const double *restrict row = aa + i * n;
            const int64_t base = i * n;
            int64_t j = 0;
            __m512i cur0 = _mm512_set_epi64(
                base + 7, base + 6, base + 5, base + 4,
                base + 3, base + 2, base + 1, base + 0);
            for (; j + 16 <= n; j += 16) {
                const __m512d v0 = _mm512_loadu_pd(row + j);
                const __m512d v1 = _mm512_loadu_pd(row + j + 8);
                __mmask8 gt0 = _mm512_cmp_pd_mask(v0, vmax0, _CMP_GT_OQ);
                __mmask8 gt1 = _mm512_cmp_pd_mask(v1, vmax1, _CMP_GT_OQ);
                vmax0 = _mm512_mask_blend_pd(gt0, vmax0, v0);
                vmax1 = _mm512_mask_blend_pd(gt1, vmax1, v1);
                vidx0 = _mm512_mask_blend_epi64(gt0, vidx0, cur0);
                __m512i cur1 = _mm512_add_epi64(cur0, inc8);
                vidx1 = _mm512_mask_blend_epi64(gt1, vidx1, cur1);
                cur0 = _mm512_add_epi64(cur1, inc8);
            }
            if (j < n) {
                const __mmask8 mask = (__mmask8)((1u << (uint32_t)(n - j)) - 1u);
                const __m512d v0 = _mm512_mask_loadu_pd(negmax, mask, row + j);
                __mmask8 gt0 = _mm512_cmp_pd_mask(v0, vmax0, _CMP_GT_OQ);
                vmax0 = _mm512_mask_blend_pd(gt0, vmax0, v0);
                vidx0 = _mm512_mask_blend_epi64(gt0, vidx0, cur0);
            }
        }

        __mmask8 gt = _mm512_cmp_pd_mask(vmax1, vmax0, _CMP_GT_OQ);
        __mmask8 eq = _mm512_cmp_pd_mask(vmax1, vmax0, _CMP_EQ_OQ);
        __mmask8 lt = _mm512_cmp_epi64_mask(vidx1, vidx0, _MM_CMPINT_LT);
        __mmask8 pick1 = gt | (eq & lt);
        vmax0 = _mm512_mask_blend_pd(pick1, vmax0, vmax1);
        vidx0 = _mm512_mask_blend_epi64(pick1, vidx0, vidx1);

        alignas(64) double maxv_buf[8];
        alignas(64) int64_t idx_buf[8];
        _mm512_storeu_pd(maxv_buf, vmax0);
        _mm512_storeu_si512((__m512i *)idx_buf, vidx0);

        double lmaxv = -DBL_MAX;
        int64_t lidx = INT64_MAX;
        for (int p = 0; p < 8; ++p) {
            const double v = maxv_buf[p];
            const int64_t k = idx_buf[p];
            if (v > lmaxv || (v == lmaxv && k < lidx)) {
                lmaxv = v;
                lidx = k;
            }
        }
        partials[tid].maxv = lmaxv;
        partials[tid].idx = lidx;
    }

    double gmaxv = partials[0].maxv;
    int64_t gidx = partials[0].idx;
    for (int t = 1; t < maxt; ++t) {
        const double v = partials[t].maxv;
        const int64_t k = partials[t].idx;
        if (v > gmaxv || (v == gmaxv && k < gidx)) {
            gmaxv = v;
            gidx = k;
        }
    }
    free(partials);

    const int64_t xindex = gidx / n;
    const int64_t yindex = gidx % n;
    bb[0] = gmaxv + (double)xindex + (double)yindex;
}
