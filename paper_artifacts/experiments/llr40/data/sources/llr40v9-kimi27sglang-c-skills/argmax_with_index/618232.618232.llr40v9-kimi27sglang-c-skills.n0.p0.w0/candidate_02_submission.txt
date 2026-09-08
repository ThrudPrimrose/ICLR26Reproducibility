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

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        out_value[0] = 0.0;
        out_index[0] = 0;
        return;
    }

    if (LEN_1D <= 256) {
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
            if (end - start >= 8) {
                __m512d vmax = _mm512_loadu_pd(&a[i]);
                __m512i vidx = _mm512_set_epi64(i + 7, i + 6, i + 5, i + 4,
                                                i + 3, i + 2, i + 1, i);
                i += 8;
                for (; i + 8 <= end; i += 8) {
                    __m512d va = _mm512_loadu_pd(&a[i]);
                    __m512i idxv = _mm512_set_epi64(i + 7, i + 6, i + 5, i + 4,
                                                    i + 3, i + 2, i + 1, i);
                    __mmask8 mask = _mm512_cmp_pd_mask(va, vmax, _CMP_GT_OQ);
                    vmax = _mm512_mask_max_pd(vmax, mask, vmax, va);
                    vidx = _mm512_mask_blend_epi64(mask, vidx, idxv);
                }

                double vals[8];
                int64_t ids[8];
                _mm512_storeu_pd(vals, vmax);
                _mm512_storeu_si512(ids, vidx);
                local.val = vals[0];
                local.idx = ids[0];
                for (int k = 1; k < 8; ++k) {
                    pair_t cand = {vals[k], ids[k]};
                    local = combine_pair(cand, local);
                }
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
