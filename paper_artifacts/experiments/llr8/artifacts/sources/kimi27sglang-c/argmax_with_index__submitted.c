#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline void reduce512(__m512d vmax, __m512i imax,
                             double *max_val, int64_t *idx)
{
    __m256d lo = _mm512_extractf64x4_pd(vmax, 0);
    __m256d hi = _mm512_extractf64x4_pd(vmax, 1);
    __m256d mx256 = _mm256_max_pd(lo, hi);
    __m128d mx128 = _mm256_extractf128_pd(mx256, 0);
    __m128d mx128_hi = _mm256_extractf128_pd(mx256, 1);
    mx128 = _mm_max_pd(mx128, mx128_hi);
    __m128d mx128_dup = _mm_unpackhi_pd(mx128, mx128);
    mx128 = _mm_max_pd(mx128, mx128_dup);
    double val = _mm_cvtsd_f64(mx128);

    __mmask8 eq_mask = _mm512_cmp_pd_mask(vmax, _mm512_set1_pd(val), _CMP_EQ_OQ);
    if (eq_mask == 0) {
        *max_val = val;
        *idx = 0;
        return;
    }
    int lane = __builtin_ctz(eq_mask);
    alignas(64) int64_t idx_buf[8];
    _mm512_store_si512((__m512i *)idx_buf, imax);
    *max_val = val;
    *idx = idx_buf[lane];
}

static void argmax_chunk(const double * restrict a,
                         int64_t start, int64_t end,
                         double * restrict oval, int64_t * restrict oidx)
{
    double max_val = a[start];
    int64_t idx = start;
    int64_t i = start + 1;

    for (; i < end && (i & 7); i++) {
        if (a[i] > max_val) {
            max_val = a[i];
            idx = i;
        }
    }

    if (i + 31 < end) {
        __m512d vmax0 = _mm512_loadu_pd(&a[i]);
        __m512d vmax1 = _mm512_loadu_pd(&a[i + 8]);
        __m512d vmax2 = _mm512_loadu_pd(&a[i + 16]);
        __m512d vmax3 = _mm512_loadu_pd(&a[i + 24]);
        __m512i imax0 = _mm512_add_epi64(_mm512_set1_epi64(i),
                                         _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
        __m512i imax1 = _mm512_add_epi64(_mm512_set1_epi64(i + 8),
                                         _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
        __m512i imax2 = _mm512_add_epi64(_mm512_set1_epi64(i + 16),
                                         _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
        __m512i imax3 = _mm512_add_epi64(_mm512_set1_epi64(i + 24),
                                         _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
        i += 32;
        for (; i + 31 < end; i += 32) {
            __m512d va0 = _mm512_loadu_pd(&a[i]);
            __m512d va1 = _mm512_loadu_pd(&a[i + 8]);
            __m512d va2 = _mm512_loadu_pd(&a[i + 16]);
            __m512d va3 = _mm512_loadu_pd(&a[i + 24]);
            __mmask8 m0 = _mm512_cmp_pd_mask(va0, vmax0, _CMP_GT_OQ);
            __mmask8 m1 = _mm512_cmp_pd_mask(va1, vmax1, _CMP_GT_OQ);
            __mmask8 m2 = _mm512_cmp_pd_mask(va2, vmax2, _CMP_GT_OQ);
            __mmask8 m3 = _mm512_cmp_pd_mask(va3, vmax3, _CMP_GT_OQ);
            vmax0 = _mm512_mask_blend_pd(m0, vmax0, va0);
            vmax1 = _mm512_mask_blend_pd(m1, vmax1, va1);
            vmax2 = _mm512_mask_blend_pd(m2, vmax2, va2);
            vmax3 = _mm512_mask_blend_pd(m3, vmax3, va3);
            __m512i vi0 = _mm512_add_epi64(_mm512_set1_epi64(i),
                                           _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
            __m512i vi1 = _mm512_add_epi64(_mm512_set1_epi64(i + 8),
                                           _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
            __m512i vi2 = _mm512_add_epi64(_mm512_set1_epi64(i + 16),
                                           _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
            __m512i vi3 = _mm512_add_epi64(_mm512_set1_epi64(i + 24),
                                           _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
            imax0 = _mm512_mask_blend_epi64(m0, imax0, vi0);
            imax1 = _mm512_mask_blend_epi64(m1, imax1, vi1);
            imax2 = _mm512_mask_blend_epi64(m2, imax2, vi2);
            imax3 = _mm512_mask_blend_epi64(m3, imax3, vi3);
        }
        // merge four vectors
        __mmask8 m;
        m = _mm512_cmp_pd_mask(vmax1, vmax0, _CMP_GT_OQ);
        vmax0 = _mm512_mask_blend_pd(m, vmax0, vmax1);
        imax0 = _mm512_mask_blend_epi64(m, imax0, imax1);
        m = _mm512_cmp_pd_mask(vmax2, vmax0, _CMP_GT_OQ);
        vmax0 = _mm512_mask_blend_pd(m, vmax0, vmax2);
        imax0 = _mm512_mask_blend_epi64(m, imax0, imax2);
        m = _mm512_cmp_pd_mask(vmax3, vmax0, _CMP_GT_OQ);
        vmax0 = _mm512_mask_blend_pd(m, vmax0, vmax3);
        imax0 = _mm512_mask_blend_epi64(m, imax0, imax3);

        double rmax;
        int64_t ridx;
        reduce512(vmax0, imax0, &rmax, &ridx);
        if (rmax > max_val) {
            max_val = rmax;
            idx = ridx;
        }
    } else if (i + 7 < end) {
        __m512d vmax = _mm512_loadu_pd(&a[i]);
        __m512i imax = _mm512_add_epi64(_mm512_set1_epi64(i),
                                         _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
        i += 8;
        for (; i + 7 < end; i += 8) {
            __m512d va = _mm512_loadu_pd(&a[i]);
            __mmask8 mask = _mm512_cmp_pd_mask(va, vmax, _CMP_GT_OQ);
            vmax = _mm512_mask_blend_pd(mask, vmax, va);
            __m512i vi = _mm512_add_epi64(_mm512_set1_epi64(i),
                                          _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
            imax = _mm512_mask_blend_epi64(mask, imax, vi);
        }
        double rmax;
        int64_t ridx;
        reduce512(vmax, imax, &rmax, &ridx);
        if (rmax > max_val) {
            max_val = rmax;
            idx = ridx;
        }
    }

    for (; i < end; i++) {
        if (a[i] > max_val) {
            max_val = a[i];
            idx = i;
        }
    }
    *oval = max_val;
    *oidx = idx;
}

void argmax_with_index_fp64(double * restrict a,
                            int64_t * restrict out_index,
                            double * restrict out_value,
                            int64_t LEN_1D,
                            uint8_t * restrict workspace,
                            int64_t workspace_bytes)
{
    if (LEN_1D <= 0) {
        *out_value = 0.0;
        *out_index = 0;
        return;
    }

    int nt = omp_get_max_threads();
    if (LEN_1D < 8 || nt <= 1 ||
        workspace == NULL ||
        workspace_bytes < nt * ((int64_t)sizeof(double) + (int64_t)sizeof(int64_t))) {
        argmax_chunk(a, 0, LEN_1D, out_value, out_index);
        return;
    }

    double *tvals = (double *)workspace;
    int64_t *tidxs = (int64_t *)(workspace + nt * sizeof(double));

    int64_t chunk = (LEN_1D + nt - 1) / nt;
    chunk = (chunk + 7) & ~7;

    #pragma omp parallel for schedule(static, 1)
    for (int t = 0; t < nt; t++) {
        int64_t start = (int64_t)t * chunk;
        if (start >= LEN_1D) {
            tvals[t] = a[0];
            tidxs[t] = 0;
        } else {
            int64_t end = start + chunk;
            if (end > LEN_1D) end = LEN_1D;
            argmax_chunk(a, start, end, &tvals[t], &tidxs[t]);
        }
    }

    double max_val = tvals[0];
    int64_t idx = tidxs[0];
    for (int t = 1; t < nt; t++) {
        if (tvals[t] > max_val) {
            max_val = tvals[t];
            idx = tidxs[t];
        }
    }
    *out_value = max_val;
    *out_index = idx;
}
