#include <stdint.h>
#include <math.h>
#include <immintrin.h>

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    const double *p = aa;

    double maxv = p[0];
    int64_t maxidx = 0;

    const __m512i offsets = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512d vmaxv = _mm512_set1_pd(-INFINITY);
    __m512i vmaxidx = _mm512_setzero_si512();

    int64_t i = 1;
    const int64_t total = n * n;

    for (; i + 8 <= total; i += 8) {
        __m512d vals = _mm512_loadu_pd(p + i);
        __m512i idxv = _mm512_add_epi64(_mm512_set1_epi64(i), offsets);
        __mmask8 gt = _mm512_cmp_pd_mask(vals, vmaxv, _CMP_GT_OQ);
        vmaxv = _mm512_mask_mov_pd(vmaxv, gt, vals);
        vmaxidx = _mm512_mask_mov_epi64(vmaxidx, gt, idxv);
    }

    /* horizontal reduce of the vector lanes */
    double vvals[8];
    int64_t vidxs[8];
    _mm512_storeu_pd(vvals, vmaxv);
    _mm512_storeu_si512((__m512i*)vidxs, vmaxidx);
    for (int k = 0; k < 8; ++k) {
        if (vvals[k] > maxv) {
            maxv = vvals[k];
            maxidx = vidxs[k];
        }
        /* ties: keep earlier (smaller) index */
    }

    for (; i < total; ++i) {
        double v = p[i];
        if (v > maxv) {
            maxv = v;
            maxidx = i;
        }
    }

    int64_t xindex = maxidx / n;
    int64_t yindex = maxidx % n;
    bb[0] = maxv + (double)xindex + (double)yindex;
}
