#include <math.h>
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

/* TSVC_2 kernel s318: max absolute value and its index, then maxv + index.
   Unroll=4, stream=True. Optimised for inc==1; other strides use parallel scalar. */

struct pair { double maxv; int64_t idx; };

static inline void maxabs_contig(const double *restrict a,
                                 int64_t start, int64_t end,
                                 double *out_maxv, int64_t *out_idx)
{
    int64_t i = start;
    int64_t n = end - start;
    double maxv;
    int64_t idx;

    if (n <= 0) {
        *out_maxv = -1.0;
        *out_idx = start;
        return;
    }

    maxv = fabs(a[start]);
    idx = start;
    i = start + 1;

    while (i < end && ((uintptr_t)&a[i] & 63)) {
        double v = fabs(a[i]);
        if (v > maxv) {
            maxv = v;
            idx = i;
        }
        ++i;
    }

    const int64_t m = (end - i) & ~((int64_t)31);
    const int64_t last = i + m;

    if (m > 0) {
        const __m512d signmask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL));
        const __m512i lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        const __m512i inc8 = _mm512_set1_epi64(8);

        __m512d maxv0 = _mm512_set1_pd(maxv);
        __m512i idx0 = _mm512_set1_epi64(idx);
        __m512d maxv1 = maxv0;
        __m512i idx1 = idx0;
        __m512d maxv2 = maxv0;
        __m512i idx2 = idx0;
        __m512d maxv3 = maxv0;
        __m512i idx3 = idx0;
        __m512i cur_idx = _mm512_add_epi64(_mm512_set1_epi64(i), lane);
        __mmask8 gt;

        for (; i < last; i += 32) {
            __m512d v0 = _mm512_and_pd(signmask, _mm512_castsi512_pd(_mm512_stream_load_si512((__m512i *)&a[i + 0])));
            gt = _mm512_cmp_pd_mask(v0, maxv0, _CMP_GT_OQ);
            maxv0 = _mm512_mask_mov_pd(maxv0, gt, v0);
            idx0 = _mm512_mask_mov_epi64(idx0, gt, cur_idx);
            cur_idx = _mm512_add_epi64(cur_idx, inc8);
            __m512d v1 = _mm512_and_pd(signmask, _mm512_castsi512_pd(_mm512_stream_load_si512((__m512i *)&a[i + 8])));
            gt = _mm512_cmp_pd_mask(v1, maxv1, _CMP_GT_OQ);
            maxv1 = _mm512_mask_mov_pd(maxv1, gt, v1);
            idx1 = _mm512_mask_mov_epi64(idx1, gt, cur_idx);
            cur_idx = _mm512_add_epi64(cur_idx, inc8);
            __m512d v2 = _mm512_and_pd(signmask, _mm512_castsi512_pd(_mm512_stream_load_si512((__m512i *)&a[i + 16])));
            gt = _mm512_cmp_pd_mask(v2, maxv2, _CMP_GT_OQ);
            maxv2 = _mm512_mask_mov_pd(maxv2, gt, v2);
            idx2 = _mm512_mask_mov_epi64(idx2, gt, cur_idx);
            cur_idx = _mm512_add_epi64(cur_idx, inc8);
            __m512d v3 = _mm512_and_pd(signmask, _mm512_castsi512_pd(_mm512_stream_load_si512((__m512i *)&a[i + 24])));
            gt = _mm512_cmp_pd_mask(v3, maxv3, _CMP_GT_OQ);
            maxv3 = _mm512_mask_mov_pd(maxv3, gt, v3);
            idx3 = _mm512_mask_mov_epi64(idx3, gt, cur_idx);
            cur_idx = _mm512_add_epi64(cur_idx, inc8);
        }

        gt = _mm512_cmp_pd_mask(maxv1, maxv0, _CMP_GT_OQ);
        maxv0 = _mm512_mask_mov_pd(maxv0, gt, maxv1);
        idx0 = _mm512_mask_mov_epi64(idx0, gt, idx1);
        gt = _mm512_cmp_pd_mask(maxv2, maxv0, _CMP_GT_OQ);
        maxv0 = _mm512_mask_mov_pd(maxv0, gt, maxv2);
        idx0 = _mm512_mask_mov_epi64(idx0, gt, idx2);
        gt = _mm512_cmp_pd_mask(maxv3, maxv0, _CMP_GT_OQ);
        maxv0 = _mm512_mask_mov_pd(maxv0, gt, maxv3);
        idx0 = _mm512_mask_mov_epi64(idx0, gt, idx3);

        double mv[8];
        int64_t iv[8];
        _mm512_storeu_pd(mv, maxv0);
        _mm512_storeu_si512(iv, idx0);
        for (int j = 0; j < 8; ++j) {
            if (mv[j] > maxv) {
                maxv = mv[j];
                idx = iv[j];
            }
        }
    }

    for (; i < end; ++i) {
        double v = fabs(a[i]);
        if (v > maxv) {
            maxv = v;
            idx = i;
        }
    }

    *out_maxv = maxv;
    *out_idx = idx;
}

static inline void maxabs_stride_chunk(const double *restrict a,
                                       int64_t start_i, int64_t end_i,
                                       int64_t inc,
                                       double *out_maxv, int64_t *out_idx)
{
    double maxv = fabs(a[start_i * inc]);
    int64_t idx = start_i;
    int64_t k = start_i * inc;
    for (int64_t i = start_i + 1; i < end_i; ++i) {
        k += inc;
        double v = fabs(a[k]);
        if (v > maxv) {
            maxv = v;
            idx = i;
        }
    }
    *out_maxv = maxv;
    *out_idx = idx;
}

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc)
{
    double maxv;
    int64_t idx;

    if (inc == 1 && LEN_1D > 1) {
        int64_t nt = omp_get_max_threads();
        if (nt > 1 && LEN_1D >= 65536) {
            struct pair local[nt];

#pragma omp parallel num_threads(nt)
            {
                int tid = omp_get_thread_num();
                int64_t chunk = LEN_1D / nt;
                int64_t rem = LEN_1D % nt;
                int64_t start = tid * chunk + (tid < rem ? tid : rem);
                int64_t end = start + chunk + (tid < rem ? 1 : 0);
                maxabs_contig(a, start, end, &local[tid].maxv, &local[tid].idx);
            }

            maxv = local[0].maxv;
            idx = local[0].idx;
            for (int64_t t = 1; t < nt; ++t) {
                if (local[t].maxv > maxv) {
                    maxv = local[t].maxv;
                    idx = local[t].idx;
                }
            }
        } else {
            maxabs_contig(a, 0, LEN_1D, &maxv, &idx);
        }
    } else {
        int64_t nt = omp_get_max_threads();
        if (nt > 1 && LEN_1D >= 65536) {
            struct pair local[nt];

#pragma omp parallel num_threads(nt)
            {
                int tid = omp_get_thread_num();
                int64_t chunk = LEN_1D / nt;
                int64_t rem = LEN_1D % nt;
                int64_t start = tid * chunk + (tid < rem ? tid : rem);
                int64_t end = start + chunk + (tid < rem ? 1 : 0);
                if (end > LEN_1D) end = LEN_1D;
                maxabs_stride_chunk(a, start, end, inc,
                                    &local[tid].maxv, &local[tid].idx);
            }

            maxv = local[0].maxv;
            idx = local[0].idx;
            for (int64_t t = 1; t < nt; ++t) {
                if (local[t].maxv > maxv) {
                    maxv = local[t].maxv;
                    idx = local[t].idx;
                }
            }
        } else {
            maxabs_stride_chunk(a, 0, LEN_1D, inc, &maxv, &idx);
        }
    }

    result[0] = maxv + (double)idx;
}

