#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

typedef struct {
    double val;
    int64_t idx;
} pair_t;

static inline pair_t combine_pair(pair_t a, pair_t b) {
    return (a.val > b.val || (a.val == b.val && a.idx < b.idx)) ? a : b;
}

static inline pair_t reduce_m512(__m512d v, __m512i vi) {
    double vals[8];
    int64_t ids[8];
    _mm512_storeu_pd(vals, v);
    _mm512_storeu_si512(ids, vi);
    pair_t best = {vals[0], ids[0]};
    for (int k = 1; k < 8; ++k) {
        pair_t cand = {vals[k], ids[k]};
        best = combine_pair(cand, best);
    }
    return best;
}

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        out_value[0] = 0.0;
        out_index[0] = 0;
        return;
    }

    /* Tiny inputs: the cost of a thread team dominates, so stay serial. */
    if (LEN_1D <= 65536) {
        pair_t best = {a[0], 0};
        for (int64_t i = 1; i < LEN_1D; ++i) {
            if (a[i] > best.val) {
                best.val = a[i];
                best.idx = i;
            }
        }
        out_value[0] = best.val;
        out_index[0] = best.idx;
        return;
    }

    pair_t global = {a[0], 0};

    #pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t base = LEN_1D / nthreads;
        const int64_t rem = LEN_1D % nthreads;
        const int64_t start = tid * base + (tid < rem ? tid : rem);
        const int64_t end = start + base + (tid < rem);

        pair_t local;
        if (end <= start) {
            local.val = -INFINITY;
            local.idx = INT64_MAX;
        } else {
            int64_t i = start;
            const int64_t len = end - start;

            if (len >= 32) {
                __m512d v0 = _mm512_loadu_pd(&a[i]);
                __m512d v1 = _mm512_loadu_pd(&a[i + 8]);
                __m512d v2 = _mm512_loadu_pd(&a[i + 16]);
                __m512d v3 = _mm512_loadu_pd(&a[i + 24]);

                __m512i id0 = _mm512_set_epi64(i + 7, i + 6, i + 5, i + 4,
                                                i + 3, i + 2, i + 1, i);
                __m512i id1 = _mm512_set_epi64(i + 15, i + 14, i + 13, i + 12,
                                                i + 11, i + 10, i + 9, i + 8);
                __m512i id2 = _mm512_set_epi64(i + 23, i + 22, i + 21, i + 20,
                                                i + 19, i + 18, i + 17, i + 16);
                __m512i id3 = _mm512_set_epi64(i + 31, i + 30, i + 29, i + 28,
                                                i + 27, i + 26, i + 25, i + 24);

                i += 32;
                for (; i + 32 <= end; i += 32) {
                    __m512d a0 = _mm512_loadu_pd(&a[i]);
                    __m512d a1 = _mm512_loadu_pd(&a[i + 8]);
                    __m512d a2 = _mm512_loadu_pd(&a[i + 16]);
                    __m512d a3 = _mm512_loadu_pd(&a[i + 24]);

                    __m512i j0 = _mm512_set_epi64(i + 7, i + 6, i + 5, i + 4,
                                                  i + 3, i + 2, i + 1, i);
                    __m512i j1 = _mm512_set_epi64(i + 15, i + 14, i + 13, i + 12,
                                                  i + 11, i + 10, i + 9, i + 8);
                    __m512i j2 = _mm512_set_epi64(i + 23, i + 22, i + 21, i + 20,
                                                  i + 19, i + 18, i + 17, i + 16);
                    __m512i j3 = _mm512_set_epi64(i + 31, i + 30, i + 29, i + 28,
                                                  i + 27, i + 26, i + 25, i + 24);

                    __mmask8 m0 = _mm512_cmp_pd_mask(a0, v0, _CMP_GT_OQ);
                    v0 = _mm512_mask_max_pd(v0, m0, v0, a0);
                    id0 = _mm512_mask_blend_epi64(m0, id0, j0);

                    __mmask8 m1 = _mm512_cmp_pd_mask(a1, v1, _CMP_GT_OQ);
                    v1 = _mm512_mask_max_pd(v1, m1, v1, a1);
                    id1 = _mm512_mask_blend_epi64(m1, id1, j1);

                    __mmask8 m2 = _mm512_cmp_pd_mask(a2, v2, _CMP_GT_OQ);
                    v2 = _mm512_mask_max_pd(v2, m2, v2, a2);
                    id2 = _mm512_mask_blend_epi64(m2, id2, j2);

                    __mmask8 m3 = _mm512_cmp_pd_mask(a3, v3, _CMP_GT_OQ);
                    v3 = _mm512_mask_max_pd(v3, m3, v3, a3);
                    id3 = _mm512_mask_blend_epi64(m3, id3, j3);
                }

                local = reduce_m512(v0, id0);
                local = combine_pair(reduce_m512(v1, id1), local);
                local = combine_pair(reduce_m512(v2, id2), local);
                local = combine_pair(reduce_m512(v3, id3), local);
            } else {
                local.val = a[i];
                local.idx = i;
                ++i;
            }

            for (; i < end; ++i) {
                if (a[i] > local.val) {
                    local.val = a[i];
                    local.idx = i;
                }
            }
        }

        #pragma omp critical
        {
            global = combine_pair(local, global);
        }
    }

    out_value[0] = global.val;
    out_index[0] = global.idx;
}
