#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <immintrin.h>
#include <omp.h>

typedef struct { double v; int64_t i; } vi;

static inline vi combine(vi a, vi b) {
    return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
}

static inline vi argmax_chunk(const double *restrict a, int64_t lo, int64_t hi) {
    vi best = {-INFINITY, INT64_MAX};
    if (hi - lo >= 4) {
        __m256d best_val = _mm256_set1_pd(-INFINITY);
        __m256i best_idx = _mm256_set1_epi64x(INT64_MAX);
        __m256i idx = _mm256_set_epi64x(lo + 3, lo + 2, lo + 1, lo + 0);
        const __m256i inc = _mm256_set1_epi64x(4);
        int64_t i = lo;
        for (; i + 4 <= hi; i += 4) {
            __m256d v = _mm256_loadu_pd(a + i);
            __m256d gt = _mm256_cmp_pd(v, best_val, _CMP_GT_OQ);
            __m256i mask = _mm256_castpd_si256(gt);
            best_val = _mm256_blendv_pd(best_val, v, gt);
            best_idx = _mm256_blendv_epi8(best_idx, idx, mask);
            idx = _mm256_add_epi64(idx, inc);
        }

        _Alignas(32) double vals[4];
        _Alignas(32) int64_t idxs[4];
        _mm256_store_pd(vals, best_val);
        _mm256_store_si256((__m256i *)idxs, best_idx);

        best.v = vals[0];
        best.i = idxs[0];
        for (int k = 1; k < 4; ++k) {
            double vk = vals[k];
            if (vk > best.v || (vk == best.v && idxs[k] < best.i)) {
                best.v = vk;
                best.i = idxs[k];
            }
        }
        for (; i < hi; ++i) {
            double ai = a[i];
            if (ai > best.v) {
                best.v = ai;
                best.i = i;
            }
        }
    } else {
        for (int64_t i = lo; i < hi; ++i) {
            double ai = a[i];
            if (ai > best.v) {
                best.v = ai;
                best.i = i;
            }
        }
    }
    return best;
}

void argmax_with_index_fp64(const double *restrict a, int64_t *restrict out_index,
                            double *restrict out_value, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;

    if (n < 4096) {
        vi best = argmax_chunk(a, 0, n);
        out_value[0] = best.v;
        out_index[0] = best.i;
        return;
    }

    int nt = omp_get_max_threads();
    if (nt > 64) nt = 64;
    vi locals[64];

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int64_t lo = (n * (int64_t)tid) / nt;
        int64_t hi = (n * (int64_t)(tid + 1)) / nt;
        locals[tid] = argmax_chunk(a, lo, hi);
    }

    vi best = locals[0];
    for (int t = 1; t < nt; ++t) best = combine(best, locals[t]);

    out_value[0] = best.v;
    out_index[0] = best.i;
}
