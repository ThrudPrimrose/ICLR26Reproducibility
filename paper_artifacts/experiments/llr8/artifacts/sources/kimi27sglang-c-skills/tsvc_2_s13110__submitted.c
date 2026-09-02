#include <stdint.h>
#include <math.h>
#include <float.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s13110_fp64(double * restrict aa, double * restrict bb, int64_t LEN_2D,
                        uint8_t * restrict workspace, int64_t workspace_bytes) {
    (void)workspace; (void)workspace_bytes;
    if (LEN_2D <= 0) return;

    const int64_t n = LEN_2D;
    const __m512d lane_off = _mm512_set_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);

    double global_max = aa[0];
    int64_t global_idx = 0;

    #pragma omp parallel
    {
        double tmax = -INFINITY;
        int64_t tidx = INT64_MAX;

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            const double * restrict row = aa + i * n;
            const int64_t base = i * n;

            __m512d vmax = _mm512_set1_pd(-INFINITY);
            __m512d vidx = _mm512_set1_pd((double)INT64_MAX);

            int64_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m512d v = _mm512_loadu_pd(row + j);
                __m512d curidx = _mm512_add_pd(_mm512_set1_pd((double)(base + j)), lane_off);
                __mmask8 gt = _mm512_cmp_pd_mask(v, vmax, _CMP_GT_OQ);
                vmax = _mm512_mask_blend_pd(gt, vmax, v);
                vidx = _mm512_mask_blend_pd(gt, vidx, curidx);
            }

            // reduce vector to scalar
            double vals[8];
            double idxs[8];
            _mm512_storeu_pd(vals, vmax);
            _mm512_storeu_pd(idxs, vidx);
            for (int k = 0; k < 8; ++k) {
                double v = vals[k];
                int64_t idx = (int64_t)idxs[k];
                if (v > tmax || (v == tmax && idx < tidx)) {
                    tmax = v;
                    tidx = idx;
                }
            }

            // scalar tail
            for (; j < n; ++j) {
                double v = row[j];
                int64_t idx = base + j;
                if (v > tmax || (v == tmax && idx < tidx)) {
                    tmax = v;
                    tidx = idx;
                }
            }
        }

        #pragma omp critical
        {
            if (tmax > global_max || (tmax == global_max && tidx < global_idx)) {
                global_max = tmax;
                global_idx = tidx;
            }
        }
    }

    int64_t xindex = global_idx / n;
    int64_t yindex = global_idx % n;
    bb[0] = global_max + (double)xindex + (double)yindex;
}
