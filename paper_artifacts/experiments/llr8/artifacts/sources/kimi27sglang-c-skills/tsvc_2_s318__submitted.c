#include <math.h>
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#define MAXT 1024

typedef struct {
    double maxv;
    int64_t idx;
} maxloc_t;

static __attribute__((noinline))
maxloc_t tsvc_2_s318_parallel_avx512(const double *restrict a,
                                        int64_t LEN_1D, double first) {
    double part_max[MAXT];
    int64_t part_idx[MAXT];

    const int nthreads = omp_get_max_threads();
    const int used = (nthreads < MAXT) ? nthreads : MAXT;
    for (int t = 0; t < used; ++t) {
        part_max[t] = -HUGE_VAL;
        part_idx[t] = -1;
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        const int64_t n = LEN_1D - 1;
        const int64_t base = n / nt;
        const int64_t rem = n % nt;
        const int64_t lo = 1 + tid * base + (tid < rem ? tid : rem);
        const int64_t hi = lo + base + (tid < rem ? 1 : 0);

        double local_max = -HUGE_VAL;
        int64_t local_idx = -1;
        int64_t i = lo;

        /* scalar peel to a 64-byte boundary */
        for (; i < hi && (((uintptr_t)(a + i)) & 63); ++i) {
            const double v = fabs(a[i]);
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        const __m512d sign_mask = _mm512_set1_pd(-0.0);
        const __m512d offset = _mm512_set_pd(7.0, 6.0, 5.0, 4.0,
                                             3.0, 2.0, 1.0, 0.0);
        __m512d best_v = _mm512_set1_pd(-HUGE_VAL);
        __m512d best_i = _mm512_setzero_pd();

        /* vector main loop: keep per-lane running maxima and earliest indices */
        for (; i + 8 <= hi; i += 8) {
            const __m512d v = _mm512_load_pd(a + i);
            const __m512d av = _mm512_andnot_pd(sign_mask, v);
            const __mmask8 gt = _mm512_cmp_pd_mask(av, best_v, _CMP_GT_OQ);
            best_v = _mm512_mask_blend_pd(gt, best_v, av);
            best_i = _mm512_mask_blend_pd(gt, best_i,
                                          _mm512_add_pd(_mm512_set1_pd((double)i), offset));
        }

        /* reduce the 8 lanes to the chunk maximum and earliest index */
        const double chunk_max = _mm512_reduce_max_pd(best_v);
        if (chunk_max > local_max) {
            const __mmask8 eq = _mm512_cmpeq_pd_mask(best_v, _mm512_set1_pd(chunk_max));
            const int lane = __builtin_ctz((unsigned)eq);
            alignas(64) double idx_buf[8];
            _mm512_store_pd(idx_buf, best_i);
            local_max = chunk_max;
            local_idx = (int64_t)idx_buf[lane];
        }

        /* scalar tail */
        for (; i < hi; ++i) {
            const double v = fabs(a[i]);
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        if (tid < used) {
            part_max[tid] = local_max;
            part_idx[tid] = local_idx;
        }
    }

    double maxv = -HUGE_VAL;
    int64_t index = -1;
    for (int t = 0; t < used; ++t) {
        const double lm = part_max[t];
        const int64_t li = part_idx[t];
        if (lm > maxv || (lm == maxv && li < index)) {
            maxv = lm;
            index = li;
        }
    }

    if (first > maxv || (first == maxv && 0 < index)) {
        maxv = first;
        index = 0;
    }

    return (maxloc_t){ .maxv = maxv, .idx = index };
}

static __attribute__((noinline))
maxloc_t tsvc_2_s318_parallel_scalar(const double *restrict a,
                                        int64_t LEN_1D, int64_t inc,
                                        double first) {
    double part_max[MAXT];
    int64_t part_idx[MAXT];

    const int nthreads = omp_get_max_threads();
    const int used = (nthreads < MAXT) ? nthreads : MAXT;
    for (int t = 0; t < used; ++t) {
        part_max[t] = -HUGE_VAL;
        part_idx[t] = -1;
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();

        const int64_t n = LEN_1D - 1;
        const int64_t base = n / nt;
        const int64_t rem = n % nt;
        const int64_t lo = 1 + tid * base + (tid < rem ? tid : rem);
        const int64_t hi = lo + base + (tid < rem ? 1 : 0);

        double local_max = -HUGE_VAL;
        int64_t local_idx = -1;

        for (int64_t i = lo; i < hi; ++i) {
            const double v = fabs(a[i * inc]);
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        if (tid < used) {
            part_max[tid] = local_max;
            part_idx[tid] = local_idx;
        }
    }

    double maxv = -HUGE_VAL;
    int64_t index = -1;
    for (int t = 0; t < used; ++t) {
        const double lm = part_max[t];
        const int64_t li = part_idx[t];
        if (lm > maxv || (lm == maxv && li < index)) {
            maxv = lm;
            index = li;
        }
    }

    if (first > maxv || (first == maxv && 0 < index)) {
        maxv = first;
        index = 0;
    }

    return (maxloc_t){ .maxv = maxv, .idx = index };
}

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result,
                      const int64_t LEN_1D, const int64_t inc) {
    if (LEN_1D <= 0) {
        result[0] = 0.0;
        return;
    }

    /* Serial reference scan for tiny inputs: avoids OpenMP launch cost. */
    if (LEN_1D <= 8192) {
        int64_t k = 0;
        int64_t index = 0;
        double maxv = fabs(a[0]);
        k += inc;
        for (int64_t i = 1; i < LEN_1D; ++i) {
            const double v = fabs(a[k]);
            if (v > maxv) {
                index = i;
                maxv = v;
            }
            k += inc;
        }
        result[0] = maxv + (double)index;
        return;
    }

    const double first = fabs(a[0]);
    if (isnan(first)) {
        result[0] = first;
        return;
    }

    maxloc_t p;
    if (inc == 1) {
        p = tsvc_2_s318_parallel_avx512(a, LEN_1D, first);
    } else {
        p = tsvc_2_s318_parallel_scalar(a, LEN_1D, inc, first);
    }

    result[0] = p.maxv + (double)p.idx;
}
