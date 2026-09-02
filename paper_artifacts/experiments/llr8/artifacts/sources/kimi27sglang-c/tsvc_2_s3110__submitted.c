#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_s3110_fp64(double *restrict aa, double *restrict bb,
                       int64_t LEN_2D, uint8_t *restrict workspace,
                       int64_t workspace_bytes) {
    double maxv = aa[0];
    int64_t xindex = 0;
    int64_t yindex = 0;

    if (LEN_2D < 16) {
        for (int64_t i = 0; i < LEN_2D; i++) {
            double *row = aa + i * LEN_2D;
            for (int64_t j = 0; j < LEN_2D; j++) {
                double val = row[j];
                if (val > maxv) {
                    maxv = val;
                    xindex = i;
                    yindex = j;
                }
            }
        }
        bb[0] = maxv + (double)xindex + (double)yindex;
        return;
    }

    const __m512i vidx_base = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i off1 = _mm512_set1_epi64(8);

    #pragma omp parallel
    {
        double local_max = aa[0];
        int64_t local_x = 0;
        int64_t local_y = 0;

        #pragma omp for nowait schedule(static)
        for (int64_t i = 0; i < LEN_2D; i++) {
            double *row = aa + i * LEN_2D;

            __m512d vmax0 = _mm512_loadu_pd(row);
            __m512d vmax1 = _mm512_loadu_pd(row + 8);
            __m512i vidx0 = vidx_base;
            __m512i vidx1 = _mm512_add_epi64(vidx_base, off1);
            int64_t j = 16;

            for (; j + 16 <= LEN_2D; j += 16) {
                __m512d v0 = _mm512_loadu_pd(row + j);
                __m512d v1 = _mm512_loadu_pd(row + j + 8);
                __m512i new_idx0 = _mm512_add_epi64(_mm512_set1_epi64(j), vidx_base);
                __m512i new_idx1 = _mm512_add_epi64(_mm512_set1_epi64(j + 8), vidx_base);
                __mmask8 mask0 = _mm512_cmp_pd_mask(v0, vmax0, _CMP_GT_OQ);
                __mmask8 mask1 = _mm512_cmp_pd_mask(v1, vmax1, _CMP_GT_OQ);
                vmax0 = _mm512_mask_blend_pd(mask0, vmax0, v0);
                vmax1 = _mm512_mask_blend_pd(mask1, vmax1, v1);
                vidx0 = _mm512_mask_blend_epi64(mask0, vidx0, new_idx0);
                vidx1 = _mm512_mask_blend_epi64(mask1, vidx1, new_idx1);
            }

            for (; j + 8 <= LEN_2D; j += 8) {
                __m512d v0 = _mm512_loadu_pd(row + j);
                __m512i new_idx0 = _mm512_add_epi64(_mm512_set1_epi64(j), vidx_base);
                __mmask8 mask0 = _mm512_cmp_pd_mask(v0, vmax0, _CMP_GT_OQ);
                vmax0 = _mm512_mask_blend_pd(mask0, vmax0, v0);
                vidx0 = _mm512_mask_blend_epi64(mask0, vidx0, new_idx0);
            }

            if (j < LEN_2D) {
                __mmask8 tail_mask = (__mmask8)((1u << (LEN_2D - j)) - 1);
                __m512d v0 = _mm512_maskz_loadu_pd(tail_mask, row + j);
                __m512i new_idx0 = _mm512_add_epi64(_mm512_set1_epi64(j), vidx_base);
                __mmask8 mask0 = _mm512_cmp_pd_mask(v0, vmax0, _CMP_GT_OQ);
                mask0 &= tail_mask;
                vmax0 = _mm512_mask_blend_pd(mask0, vmax0, v0);
                vidx0 = _mm512_mask_blend_epi64(mask0, vidx0, new_idx0);
            }

            // combine vmax1 into vmax0
            __mmask8 mask01 = _mm512_cmp_pd_mask(vmax1, vmax0, _CMP_GT_OQ);
            vmax0 = _mm512_mask_blend_pd(mask01, vmax0, vmax1);
            vidx0 = _mm512_mask_blend_epi64(mask01, vidx0, vidx1);

            double lanes_max[8];
            int64_t lanes_idx[8];
            _mm512_storeu_pd(lanes_max, vmax0);
            _mm512_storeu_si512(lanes_idx, vidx0);

            double row_max = lanes_max[0];
            int64_t row_y = lanes_idx[0];
            for (int k = 1; k < 8; k++) {
                if (lanes_max[k] > row_max) {
                    row_max = lanes_max[k];
                    row_y = lanes_idx[k];
                }
            }

            if (row_max > local_max) {
                local_max = row_max;
                local_x = i;
                local_y = row_y;
            }
        }

        #pragma omp critical
        {
            if (local_max > maxv) {
                maxv = local_max;
                xindex = local_x;
                yindex = local_y;
            }
        }
    }

    bb[0] = maxv + (double)xindex + (double)yindex;
}
