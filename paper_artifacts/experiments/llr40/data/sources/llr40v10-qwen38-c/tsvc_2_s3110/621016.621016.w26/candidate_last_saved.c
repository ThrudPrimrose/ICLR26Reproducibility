#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#if defined(__AVX512F__)
#  include <immintrin.h>
#  define HAVE512 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define HAVE256 1
#else
#  define HAVE256 0
#endif

static inline void comb_best(double *best, int64_t *bestlin, double v, int64_t lin) {
    if (v > *best || (v == *best && lin < *bestlin)) {
        *best = v;
        *bestlin = lin;
    }
}

#if HAVE512
static void reduce512(const double *restrict x, int64_t n, double *pm, int64_t *pl) {
    if (n <= 8) {
        double m = x[0];
        int64_t l = 0;
        for (int64_t i = 1; i < n; ++i) {
            double v = x[i];
            if (v > m) { m = v; l = i; }
        }
        *pm = m; *pl = l;
        return;
    }
    __m512d m = _mm512_set1_pd(x[0]);
    __m512i idx07 = _mm512_setr_epi64(0LL, 1LL, 2LL, 3LL, 4LL, 5LL, 6LL, 7LL);
    __m512i idx = idx07;
    const int64_t nv = n & ~7LL;
    for (int64_t i = 8; i < nv; i += 8) {
        __m512d v = _mm512_loadu_pd(x + i);
        __m512d gt = _mm512_cmp_pd(v, m, _CMP_GT_OQ);
        __m512d eq = _mm512_cmp_pd(v, m, _CMP_EQ_OQ);
        m = _mm512_max_pd(m, v);
        __m512i lane = _mm512_add_epi64(idx07, _mm512_set1_epi64(i));
        idx = _mm512_blendv_epi64(idx, lane, _mm512_castpd_si512(gt));
        __m512i mn = _mm512_min_epi64(idx, lane);
        idx = _mm512_blendv_epi64(idx, mn, _mm512_castpd_si512(eq));
    }
    double tv[8];
    int64_t ti[8];
    _mm512_store_pd(tv, m);
    _mm512_store_si512((__m512i *)ti, idx);
    double m2 = tv[0];
    int64_t l2 = ti[0];
    for (int k = 1; k < 8; ++k) comb_best(&m2, &l2, tv[k], ti[k]);
    for (int64_t i = nv; i < n; ++i) {
        double v = x[i];
        if (v > m2) { m2 = v; l2 = i; }
    }
    *pm = m2; *pl = l2;
}
#elif HAVE256
static void reduce256(const double *restrict x, int64_t n, double *pm, int64_t *pl) {
    if (n <= 4) {
        double m = x[0];
        int64_t l = 0;
        for (int64_t i = 1; i < n; ++i) {
            double v = x[i];
            if (v > m) { m = v; l = i; }
        }
        *pm = m; *pl = l;
        return;
    }
    __m256d m = _mm256_set1_pd(x[0]);
    __m256i idx03 = _mm256_setr_epi64(0LL, 1LL, 2LL, 3LL);
    __m256i idx = idx03;
    const int64_t nv = n & ~3LL;
    for (int64_t i = 4; i < nv; i += 4) {
        __m256d v = _mm256_loadu_pd(x + i);
        __m256d gt = _mm256_cmp_pd(v, m, _CMP_GT_OQ);
        __m256d eq = _mm256_cmp_pd(v, m, _CMP_EQ_OQ);
        m = _mm256_max_pd(m, v);
        __m256i lane = _mm256_add_epi64(idx03, _mm256_set1_epi64(i));
        idx = _mm256_blendv_epi64(idx, lane, _mm256_castpd_si256(gt));
        __m256i mn = _mm256_min_epi64(idx, lane);
        idx = _mm256_blendv_epi64(idx, mn, _mm256_castpd_si256(eq));
    }
    double tv[4];
    int64_t ti[4];
    _mm256_store_pd(tv, m);
    _mm256_store_si256((__m256i *)ti, idx);
    double m2 = tv[0];
    int64_t l2 = ti[0];
    for (int k = 1; k < 4; ++k) comb_best(&m2, &l2, tv[k], ti[k]);
    for (int64_t i = nv; i < n; ++i) {
        double v = x[i];
        if (v > m2) { m2 = v; l2 = i; }
    }
    *pm = m2; *pl = l2;
}
#else
static void reduce_scalar(const double *restrict x, int64_t n, double *pm, int64_t *pl) {
    double m = x[0];
    int64_t l = 0;
    for (int64_t i = 1; i < n; ++i) {
        double v = x[i];
        if (v > m) { m = v; l = i; }
    }
    *pm = m; *pl = l;
}
#endif

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t n = LEN_2D * LEN_2D;
    if (n <= 0) { bb[0] = 0.0; return; }
    double best = aa[0];
    int64_t bestlin = 0;

    int nt = omp_get_max_threads();
    if (nt < 1) nt = 1;

    if (n < (1 << 17) || nt == 1) {
#if HAVE512
        reduce512(aa, n, &best, &bestlin);
#elif HAVE256
        reduce256(aa, n, &best, &bestlin);
#else
        reduce_scalar(aa, n, &best, &bestlin);
#endif
    } else {
        #pragma omp parallel num_threads(nt)
        {
            const int nthr = omp_get_num_threads();
            const int tid = omp_get_thread_num();
            const int64_t per = n / nthr;
            const int64_t b0 = (int64_t)tid * per + (int64_t)tid * (n - per * nthr);
            const int64_t b1 = b0 + per + (tid < (int)(n - per * nthr));
            double pm;
            int64_t pl;
#if HAVE512
            reduce512(aa + b0, b1 - b0, &pm, &pl);
#elif HAVE256
            reduce256(aa + b0, b1 - b0, &pm, &pl);
#else
            reduce_scalar(aa + b0, b1 - b0, &pm, &pl);
#endif
            pl += b0;
            #pragma omp critical
            comb_best(&best, &bestlin, pm, pl);
        }
    }
    bb[0] = best + (double)(bestlin / LEN_2D) + (double)(bestlin % LEN_2D);
}
