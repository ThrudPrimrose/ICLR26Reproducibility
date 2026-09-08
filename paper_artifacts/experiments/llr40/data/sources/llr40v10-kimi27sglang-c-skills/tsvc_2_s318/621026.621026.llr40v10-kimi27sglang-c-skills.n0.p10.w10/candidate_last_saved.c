#include <math.h>
#include <stdint.h>
#include <omp.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

typedef struct {
    double value;
    int64_t index;
} maxloc_t;

static inline maxloc_t combine(maxloc_t a, maxloc_t b) {
    if (b.value > a.value || (b.value == a.value && b.index < a.index))
        return b;
    return a;
}

#ifdef __AVX512F__
static inline maxloc_t maxloc_stride_one(const double *restrict a,
                                         int64_t start, int64_t end) {
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    const __m512d neg_inf = _mm512_set1_pd(-HUGE_VAL);

    __m512d maxv = neg_inf;
    __m512d idxv = _mm512_set1_pd(0.0); /* only lanes where maxv is real are read */

    int64_t i = start;
    for (; i + 8 <= end; i += 8) {
        __m512d v = _mm512_loadu_pd(&a[i]);
        v = _mm512_andnot_pd(sign_mask, v);
        __mmask8 gt = _mm512_cmp_pd_mask(v, maxv, _CMP_GT_OQ);
        maxv = _mm512_max_pd(maxv, v);
        __m512d iv = _mm512_set_pd(i + 7, i + 6, i + 5, i + 4,
                                   i + 3, i + 2, i + 1, i);
        idxv = _mm512_mask_blend_pd(gt, idxv, iv);
    }

    double lanes[8];
    int64_t lanes_i[8];
    _mm512_storeu_pd(lanes, maxv);
    _mm512_storeu_pd((double *)lanes_i, idxv); /* idx fits in int64_t */

    maxloc_t best = {-HUGE_VAL, INT64_MAX};
    for (int l = 0; l < 8; ++l) {
        maxloc_t cur = {lanes[l], lanes_i[l]};
        best = combine(best, cur);
    }

    for (; i < end; ++i) {
        maxloc_t cur = {fabs(a[i]), i};
        best = combine(best, cur);
    }
    return best;
}
#else
static inline maxloc_t maxloc_stride_one(const double *restrict a,
                                         int64_t start, int64_t end) {
    maxloc_t best = {-HUGE_VAL, INT64_MAX};
    for (int64_t i = start; i < end; ++i) {
        maxloc_t cur = {fabs(a[i]), i};
        best = combine(best, cur);
    }
    return best;
}
#endif

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc) {
    const int max_threads = 256;
    maxloc_t partials[max_threads];
    int nthreads_used = 1;

    if (inc == 1) {
        #pragma omp parallel if (LEN_1D > 4096)
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            #pragma omp single
            nthreads_used = nthreads;

            const int64_t chunk = LEN_1D / nthreads;
            const int64_t rem = LEN_1D % nthreads;
            const int64_t start = tid * chunk + (tid < rem ? tid : rem);
            const int64_t count = chunk + (tid < rem ? 1 : 0);

            partials[tid] = maxloc_stride_one(a, start, start + count);
        }
    } else {
        #pragma omp parallel if (LEN_1D > 4096)
        {
            const int tid = omp_get_thread_num();
            const int nthreads = omp_get_num_threads();
            #pragma omp single
            nthreads_used = nthreads;

            const int64_t chunk = LEN_1D / nthreads;
            const int64_t rem = LEN_1D % nthreads;
            const int64_t start = tid * chunk + (tid < rem ? tid : rem);
            const int64_t count = chunk + (tid < rem ? 1 : 0);
            const int64_t end = start + count;

            maxloc_t best = {-HUGE_VAL, INT64_MAX};
            for (int64_t i = start; i < end; ++i) {
                maxloc_t cur = {fabs(a[i * inc]), i};
                best = combine(best, cur);
            }
            partials[tid] = best;
        }
    }

    maxloc_t best = {-HUGE_VAL, INT64_MAX};
    for (int t = 0; t < nthreads_used; ++t) {
        best = combine(best, partials[t]);
    }

    result[0] = best.value + (double)best.index;
}
