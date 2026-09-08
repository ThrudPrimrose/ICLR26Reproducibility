#include <stdint.h>
#include <float.h>
#include <immintrin.h>

static inline void reduce_vmax_vidx(__m512d vmax, __m512i vidx, double *out_max, int64_t *out_idx) {
    double maxv[8];
    int64_t idxv[8];
    _mm512_storeu_pd(maxv, vmax);
    _mm512_storeu_si512((__m512i *)idxv, vidx);

    double m = maxv[0];
    int64_t ix = idxv[0];
    for (int i = 1; i < 8; ++i) {
        if (maxv[i] > m || (maxv[i] == m && idxv[i] < ix)) {
            m = maxv[i];
            ix = idxv[i];
        }
    }
    *out_max = m;
    *out_idx = ix;
}

static inline void merge_maxidx(double *m, int64_t *i, double vm, int64_t vi) {
    if (vm > *m || (vm == *m && vi < *i)) {
        *m = vm;
        *i = vi;
    }
}

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t n = LEN_2D * LEN_2D;

    const __m512d neg_inf = _mm512_set1_pd(-(DBL_MAX));
    const __m512i big_idx = _mm512_set1_epi64(INT64_MAX);
    const __m512i off = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    const __m512i eight = _mm512_set1_epi64(8);

    __m512d tmax0 = neg_inf, tmax1 = neg_inf;
    __m512i tidx0 = big_idx, tidx1 = big_idx;

    int64_t k;
    for (k = 0; k < n - 15; k += 16) {
        __m512d v0 = _mm512_loadu_pd(aa + k);
        __m512d v1 = _mm512_loadu_pd(aa + k + 8);
        __mmask8 m0 = _mm512_cmp_pd_mask(v0, tmax0, _CMP_GT_OQ);
        __mmask8 m1 = _mm512_cmp_pd_mask(v1, tmax1, _CMP_GT_OQ);
        tmax0 = _mm512_mask_blend_pd(m0, tmax0, v0);
        tmax1 = _mm512_mask_blend_pd(m1, tmax1, v1);
        __m512i base = _mm512_set1_epi64(k);
        __m512i cur0 = _mm512_add_epi64(base, off);
        __m512i cur1 = _mm512_add_epi64(cur0, eight);
        tidx0 = _mm512_mask_blend_epi64(m0, tidx0, cur0);
        tidx1 = _mm512_mask_blend_epi64(m1, tidx1, cur1);
    }

    double lmax = -(DBL_MAX);
    int64_t lidx = INT64_MAX;
    reduce_vmax_vidx(tmax0, tidx0, &lmax, &lidx);
    double lmax1;
    int64_t lidx1;
    reduce_vmax_vidx(tmax1, tidx1, &lmax1, &lidx1);
    merge_maxidx(&lmax, &lidx, lmax1, lidx1);

    for (; k < n; ++k) {
        double v = aa[k];
        if (v > lmax || (v == lmax && k < lidx)) {
            lmax = v;
            lidx = k;
        }
    }

    int64_t xindex = lidx / LEN_2D;
    int64_t yindex = lidx % LEN_2D;
    bb[0] = lmax + (double)xindex + (double)yindex;
}
