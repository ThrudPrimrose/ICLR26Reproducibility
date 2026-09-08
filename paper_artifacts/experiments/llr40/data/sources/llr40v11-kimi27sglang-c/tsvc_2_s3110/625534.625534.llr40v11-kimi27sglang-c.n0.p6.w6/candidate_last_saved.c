#include <stdint.h>
#include <immintrin.h>
#include <math.h>

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t n = LEN_2D * LEN_2D;
    double maxv;
    int64_t best_idx;

    if (n < 16) {
        maxv = aa[0];
        best_idx = 0;
        for (int64_t k = 1; k < n; ++k) {
            double v = aa[k];
            if (v > maxv) {
                maxv = v;
                best_idx = k;
            }
        }
    } else {
        const __m512i lane_idx = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        __m512d vmax0 = _mm512_loadu_pd(aa);
        __m512d vmax1 = _mm512_loadu_pd(aa + 8);
        __m512i imax0 = lane_idx;
        __m512i imax1 = _mm512_add_epi64(lane_idx, _mm512_set1_epi64(8));

        int64_t k = 16;
        for (; k + 16 <= n; k += 16) {
            __m512d v0 = _mm512_loadu_pd(aa + k);
            __m512d v1 = _mm512_loadu_pd(aa + k + 8);
            __m512i idx0 = _mm512_add_epi64(lane_idx, _mm512_set1_epi64(k));
            __m512i idx1 = _mm512_add_epi64(lane_idx, _mm512_set1_epi64(k + 8));

            __mmask8 m0 = _mm512_cmp_pd_mask(v0, vmax0, _CMP_GT_OQ);
            __mmask8 m1 = _mm512_cmp_pd_mask(v1, vmax1, _CMP_GT_OQ);

            vmax0 = _mm512_mask_blend_pd(m0, vmax0, v0);
            vmax1 = _mm512_mask_blend_pd(m1, vmax1, v1);
            imax0 = _mm512_mask_blend_epi64(m0, imax0, idx0);
            imax1 = _mm512_mask_blend_epi64(m1, imax1, idx1);
        }

        __mmask8 m01 = _mm512_cmp_pd_mask(vmax1, vmax0, _CMP_GT_OQ);
        __m512d vmax = _mm512_mask_blend_pd(m01, vmax0, vmax1);
        __m512i imax = _mm512_mask_blend_epi64(m01, imax0, imax1);

        for (; k + 8 <= n; k += 8) {
            __m512d v = _mm512_loadu_pd(aa + k);
            __m512i idxvec = _mm512_add_epi64(lane_idx, _mm512_set1_epi64(k));
            __mmask8 mask = _mm512_cmp_pd_mask(v, vmax, _CMP_GT_OQ);
            vmax = _mm512_mask_blend_pd(mask, vmax, v);
            imax = _mm512_mask_blend_epi64(mask, imax, idxvec);
        }

        double lanes[8];
        int64_t idxs[8];
        _mm512_storeu_pd(lanes, vmax);
        _mm512_storeu_si512(idxs, imax);

        maxv = lanes[0];
        best_idx = idxs[0];
        for (int i = 1; i < 8; ++i) {
            if (lanes[i] > maxv) {
                maxv = lanes[i];
                best_idx = idxs[i];
            }
        }

        for (; k < n; ++k) {
            double v = aa[k];
            if (v > maxv) {
                maxv = v;
                best_idx = k;
            }
        }
    }

    const int64_t xindex = best_idx / LEN_2D;
    const int64_t yindex = best_idx % LEN_2D;
    bb[0] = maxv + (double)xindex + (double)yindex;
}
