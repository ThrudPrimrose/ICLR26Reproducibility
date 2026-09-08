#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

struct pair { double val; int64_t idx; };

static inline void update_pair(struct pair *p, double v, int64_t i) {
    if (v > p->val) {
        p->val = v;
        p->idx = i;
    }
}

static inline struct pair argmax_range(const double *restrict a, int64_t start, int64_t end, struct pair local_best) {
    int64_t simd_start = (start + 8) & ~7;
    if (simd_start > end) simd_start = end;

    int64_t i;
    for (i = start + 1; i < simd_start; ++i) {
        if (a[i] > local_best.val) {
            local_best.val = a[i];
            local_best.idx = i;
        }
    }

    if (simd_start + 7 < end) {
        __m512d best_v = _mm512_set1_pd(local_best.val);
        __m512i best_i = _mm512_set1_epi64(local_best.idx);
        __m512i lane_offsets = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);

        for (i = simd_start; i + 7 < end; i += 8) {
            __m512d v = _mm512_loadu_pd(&a[i]);
            __mmask8 mask = _mm512_cmp_pd_mask(v, best_v, _CMP_GT_OQ);
            __m512i idx = _mm512_add_epi64(_mm512_set1_epi64(i), lane_offsets);
            best_v = _mm512_mask_blend_pd(mask, best_v, v);
            best_i = _mm512_mask_blend_epi64(mask, best_i, idx);
        }

        double vals[8];
        int64_t idxs[8];
        _mm512_storeu_pd(vals, best_v);
        _mm512_storeu_si512((__m512i *)idxs, best_i);
        for (int k = 0; k < 8; ++k) {
            if (vals[k] > local_best.val || (vals[k] == local_best.val && idxs[k] < local_best.idx)) {
                local_best.val = vals[k];
                local_best.idx = idxs[k];
            }
        }
    } else {
        i = simd_start;
    }

    for (; i < end; ++i) {
        if (a[i] > local_best.val) {
            local_best.val = a[i];
            local_best.idx = i;
        }
    }

    return local_best;
}

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index, double *restrict out_value,
                            const int64_t LEN_1D) {
    if (LEN_1D <= 0) {
        out_value[0] = 0.0;
        out_index[0] = 0;
        return;
    }

    struct pair best = { a[0], 0 };
    if (LEN_1D == 1) {
        out_value[0] = best.val;
        out_index[0] = best.idx;
        return;
    }

    const int64_t PARALLEL_THRESHOLD = 8192;

    if (LEN_1D <= PARALLEL_THRESHOLD) {
        best = argmax_range(a, 0, LEN_1D, best);
    } else {
        int max_threads = omp_get_max_threads();
        struct pair locals[max_threads];
        int nthreads = 1;
        #pragma omp parallel num_threads(max_threads)
        {
            int tid = omp_get_thread_num();
            if (tid == 0) nthreads = omp_get_num_threads();
            int nt = omp_get_num_threads();
            int64_t chunk = LEN_1D / nt;
            int64_t rem = LEN_1D % nt;
            int64_t start = tid * chunk + (tid < rem ? tid : rem);
            int64_t end = start + chunk + (tid < rem ? 1 : 0);
            struct pair local_best = { a[start], start };
            local_best = argmax_range(a, start, end, local_best);
            locals[tid] = local_best;
        }
        best = locals[0];
        for (int t = 1; t < nthreads; ++t) {
            if (locals[t].val > best.val || (locals[t].val == best.val && locals[t].idx < best.idx)) {
                best = locals[t];
            }
        }
    }

    out_value[0] = best.val;
    out_index[0] = best.idx;
}
