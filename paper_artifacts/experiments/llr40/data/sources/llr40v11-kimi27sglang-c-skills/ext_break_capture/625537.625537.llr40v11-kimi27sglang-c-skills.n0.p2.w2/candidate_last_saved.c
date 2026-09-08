#include <stdint.h>
#include <limits.h>
#include <immintrin.h>
#include <omp.h>

static inline void find_first_serial(const double *restrict a, int64_t *restrict out_index,
                                     double *restrict out_value, int64_t LEN_1D, double k) {
    const __m512i vki = _mm512_set1_epi64(INT64_C(0x3FF0000000000000));
    const __m512i lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    int64_t i = 0;
    const int64_t n32 = LEN_1D & ~31;
    for (; i < n32; i += 32) {
        __m512i a0 = _mm512_loadu_si512((const void *)(a + i + 0));
        __m512i a1 = _mm512_loadu_si512((const void *)(a + i + 8));
        __mmask8 m0 = _mm512_cmpgt_epi64_mask(a0, vki);
        __mmask8 m1 = _mm512_cmpgt_epi64_mask(a1, vki);
        if (m0 | m1) {
            int j;
            if (m0) {
                j = __builtin_ctz(m0);
            } else {
                j = 8 + __builtin_ctz(m1);
            }
            out_index[0] = i + j;
            out_value[0] = a[i + j];
            return;
        }
    }
    const int64_t n8 = LEN_1D & ~7;
    for (; i < n8; i += 8) {
        __m512i ai = _mm512_loadu_si512((const void *)(a + i));
        __mmask8 m = _mm512_cmpgt_epi64_mask(ai, vki);
        if (m) {
            int j = __builtin_ctz(m);
            out_index[0] = i + j;
            out_value[0] = a[i + j];
            return;
        }
    }
    for (; i < LEN_1D; ++i) {
        if (a[i] > k) {
            out_index[0] = i;
            out_value[0] = a[i];
            return;
        }
    }
}

void ext_break_capture_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    const double k = 1.0;
    out_index[0] = -1;
    out_value[0] = -1.0;

    if (LEN_1D <= 0) return;

    int nthreads = omp_get_max_threads();
    if (nthreads <= 1 || LEN_1D < (1LL << 17)) {
        find_first_serial(a, out_index, out_value, LEN_1D, k);
        return;
    }

    const __m512i vki = _mm512_set1_epi64(INT64_C(0x3FF0000000000000));
    const __m512i inf = _mm512_set1_epi64(INT64_MAX);
    const __m512i lane = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i eight = _mm512_set1_epi64(8);

    int64_t best = INT64_MAX;
    #pragma omp parallel reduction(min:best)
    {
        int tid = omp_get_thread_num();
        int nthr = omp_get_num_threads();
        int64_t start = (LEN_1D * tid) / nthr;
        int64_t end = (LEN_1D * (tid + 1LL)) / nthr;

        __m512i bestv = inf;
        int64_t i = start;
        int64_t n64_end = (end & ~63);

        for (; i < n64_end; i += 64) {
            __m512i idx = _mm512_add_epi64(_mm512_set1_epi64(i), lane);
            for (int v = 0; v < 8; ++v) {
                __m512i ai = _mm512_loadu_si512((const void *)(a + i + 8 * v));
                __mmask8 m = _mm512_cmpgt_epi64_mask(ai, vki);
                __m512i cand = _mm512_mask_blend_epi64(m, inf, idx);
                bestv = _mm512_min_epi64(bestv, cand);
                idx = _mm512_add_epi64(idx, eight);
            }
        }

        int64_t n8_end = end & ~7;
        for (; i < n8_end; i += 8) {
            __m512i ai = _mm512_loadu_si512((const void *)(a + i));
            __mmask8 m = _mm512_cmpgt_epi64_mask(ai, vki);
            __m512i idx = _mm512_add_epi64(_mm512_set1_epi64(i), lane);
            __m512i cand = _mm512_mask_blend_epi64(m, inf, idx);
            bestv = _mm512_min_epi64(bestv, cand);
        }

        int64_t loc = _mm512_reduce_min_epi64(bestv);
        for (; i < end; ++i) {
            if (a[i] > k && i < loc) loc = i;
        }
        if (loc < best) best = loc;
    }

    if (best < LEN_1D) {
        out_index[0] = best;
        out_value[0] = a[best];
    }
}
