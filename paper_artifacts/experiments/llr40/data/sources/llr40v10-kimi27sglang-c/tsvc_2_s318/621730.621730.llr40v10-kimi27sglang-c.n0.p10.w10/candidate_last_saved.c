#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#define USE_AVX512
#elif defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2
#endif

typedef struct {
    double maxv;
    int64_t index;
} loc_t;

static inline loc_t loc_combine(loc_t a, loc_t b) {
    if (a.maxv > b.maxv) return a;
    if (a.maxv < b.maxv) return b;
    return (a.index < b.index) ? a : b;
}

#if defined(USE_AVX512)

static inline loc_t argmax_range_avx512_contig(const double *a, int64_t start, int64_t end) {
    double maxv0 = fabs(a[0]);
    __m512d vmax0 = _mm512_set1_pd(maxv0);
    __m512d vmax1 = _mm512_set1_pd(maxv0);
    __m512i imax0 = _mm512_set1_epi64(0);
    __m512i imax1 = _mm512_set1_epi64(0);
    __m512i vi0 = _mm512_setr_epi64(start, start + 1, start + 2, start + 3,
                                    start + 4, start + 5, start + 6, start + 7);
    __m512i vi1 = _mm512_setr_epi64(start + 8, start + 9, start + 10, start + 11,
                                    start + 12, start + 13, start + 14, start + 15);
    __m512i vi_inc16 = _mm512_set1_epi64(16);
    __m512d sign_mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL));

    int64_t i;
    for (i = start; i + 15 < end; i += 16) {
        __m512d v0 = _mm512_loadu_pd(a + i);
        __m512d v1 = _mm512_loadu_pd(a + i + 8);
        v0 = _mm512_and_pd(v0, sign_mask);
        v1 = _mm512_and_pd(v1, sign_mask);
        __mmask8 k0 = _mm512_cmp_pd_mask(v0, vmax0, _CMP_GT_OQ);
        __mmask8 k1 = _mm512_cmp_pd_mask(v1, vmax1, _CMP_GT_OQ);
        vmax0 = _mm512_mask_blend_pd(k0, vmax0, v0);
        vmax1 = _mm512_mask_blend_pd(k1, vmax1, v1);
        imax0 = _mm512_mask_blend_epi64(k0, imax0, vi0);
        imax1 = _mm512_mask_blend_epi64(k1, imax1, vi1);
        vi0 = _mm512_add_epi64(vi0, vi_inc16);
        vi1 = _mm512_add_epi64(vi1, vi_inc16);
    }

    __mmask8 kc = _mm512_cmp_pd_mask(vmax1, vmax0, _CMP_GT_OQ);
    vmax0 = _mm512_mask_blend_pd(kc, vmax0, vmax1);
    imax0 = _mm512_mask_blend_epi64(kc, imax0, imax1);

    for (; i + 7 < end; i += 8) {
        __m512d v = _mm512_loadu_pd(a + i);
        v = _mm512_and_pd(v, sign_mask);
        __mmask8 k = _mm512_cmp_pd_mask(v, vmax0, _CMP_GT_OQ);
        vmax0 = _mm512_mask_blend_pd(k, vmax0, v);
        imax0 = _mm512_mask_blend_epi64(k, imax0, vi0);
        vi0 = _mm512_add_epi64(vi0, _mm512_set1_epi64(8));
    }

    __attribute__((aligned(64))) double mv[8];
    __attribute__((aligned(64))) int64_t iv[8];
    _mm512_store_pd(mv, vmax0);
    _mm512_store_si512((__m512i *)iv, imax0);

    loc_t loc = {mv[0], iv[0]};
    for (int j = 1; j < 8; ++j) {
        if (mv[j] > loc.maxv || (mv[j] == loc.maxv && iv[j] < loc.index)) {
            loc.maxv = mv[j];
            loc.index = iv[j];
        }
    }

    for (; i < end; ++i) {
        double v = fabs(a[i]);
        if (v > loc.maxv || (v == loc.maxv && i < loc.index)) {
            loc.maxv = v;
            loc.index = i;
        }
    }
    return loc;
}

static inline loc_t argmax_range_avx512_strided(const double *a, int64_t start, int64_t end, int64_t inc) {
    double maxv0 = fabs(a[0]);
    __m512d vmax0 = _mm512_set1_pd(maxv0);
    __m512d vmax1 = _mm512_set1_pd(maxv0);
    __m512i imax0 = _mm512_set1_epi64(0);
    __m512i imax1 = _mm512_set1_epi64(0);
    __m512i vidx0 = _mm512_setr_epi64(start * inc, (start + 1) * inc, (start + 2) * inc, (start + 3) * inc,
                                      (start + 4) * inc, (start + 5) * inc, (start + 6) * inc, (start + 7) * inc);
    __m512i vidx1 = _mm512_setr_epi64((start + 8) * inc, (start + 9) * inc, (start + 10) * inc, (start + 11) * inc,
                                      (start + 12) * inc, (start + 13) * inc, (start + 14) * inc, (start + 15) * inc);
    __m512i vi0 = _mm512_setr_epi64(start, start + 1, start + 2, start + 3,
                                    start + 4, start + 5, start + 6, start + 7);
    __m512i vi1 = _mm512_setr_epi64(start + 8, start + 9, start + 10, start + 11,
                                    start + 12, start + 13, start + 14, start + 15);
    __m512i vidx_inc16 = _mm512_set1_epi64(16 * inc);
    __m512i vi_inc16 = _mm512_set1_epi64(16);
    __m512d sign_mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL));

    int64_t i;
    for (i = start; i + 15 < end; i += 16) {
        _mm_prefetch((const char *)(a + (i + 64) * inc), _MM_HINT_T0);
        _mm_prefetch((const char *)(a + (i + 64 + 8) * inc), _MM_HINT_T0);
        __m512d v0 = _mm512_i64gather_pd(vidx0, a, 8);
        __m512d v1 = _mm512_i64gather_pd(vidx1, a, 8);
        v0 = _mm512_and_pd(v0, sign_mask);
        v1 = _mm512_and_pd(v1, sign_mask);
        __mmask8 k0 = _mm512_cmp_pd_mask(v0, vmax0, _CMP_GT_OQ);
        __mmask8 k1 = _mm512_cmp_pd_mask(v1, vmax1, _CMP_GT_OQ);
        vmax0 = _mm512_mask_blend_pd(k0, vmax0, v0);
        vmax1 = _mm512_mask_blend_pd(k1, vmax1, v1);
        imax0 = _mm512_mask_blend_epi64(k0, imax0, vi0);
        imax1 = _mm512_mask_blend_epi64(k1, imax1, vi1);
        vidx0 = _mm512_add_epi64(vidx0, vidx_inc16);
        vidx1 = _mm512_add_epi64(vidx1, vidx_inc16);
        vi0 = _mm512_add_epi64(vi0, vi_inc16);
        vi1 = _mm512_add_epi64(vi1, vi_inc16);
    }

    __mmask8 kc = _mm512_cmp_pd_mask(vmax1, vmax0, _CMP_GT_OQ);
    vmax0 = _mm512_mask_blend_pd(kc, vmax0, vmax1);
    imax0 = _mm512_mask_blend_epi64(kc, imax0, imax1);

    for (; i + 7 < end; i += 8) {
        __m512d v = _mm512_i64gather_pd(vidx0, a, 8);
        v = _mm512_and_pd(v, sign_mask);
        __mmask8 k = _mm512_cmp_pd_mask(v, vmax0, _CMP_GT_OQ);
        vmax0 = _mm512_mask_blend_pd(k, vmax0, v);
        imax0 = _mm512_mask_blend_epi64(k, imax0, vi0);
        vidx0 = _mm512_add_epi64(vidx0, _mm512_set1_epi64(8 * inc));
        vi0 = _mm512_add_epi64(vi0, _mm512_set1_epi64(8));
    }

    __attribute__((aligned(64))) double mv[8];
    __attribute__((aligned(64))) int64_t iv[8];
    _mm512_store_pd(mv, vmax0);
    _mm512_store_si512((__m512i *)iv, imax0);

    loc_t loc = {mv[0], iv[0]};
    for (int j = 1; j < 8; ++j) {
        if (mv[j] > loc.maxv || (mv[j] == loc.maxv && iv[j] < loc.index)) {
            loc.maxv = mv[j];
            loc.index = iv[j];
        }
    }

    int64_t k = i * inc;
    for (; i < end; ++i) {
        double v = fabs(a[k]);
        if (v > loc.maxv || (v == loc.maxv && i < loc.index)) {
            loc.maxv = v;
            loc.index = i;
        }
        k += inc;
    }
    return loc;
}

#elif defined(USE_AVX2)

static inline loc_t argmax_range_avx2_contig(const double *a, int64_t start, int64_t end) {
    double maxv0 = fabs(a[0]);
    __m256d vmax = _mm256_set1_pd(maxv0);
    __m256i imax = _mm256_set1_epi64x(0);
    __m256i vi = _mm256_setr_epi64x(start, start + 1, start + 2, start + 3);
    __m256i vi_inc = _mm256_set1_epi64x(4);
    __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));

    int64_t i;
    for (i = start; i + 3 < end; i += 4) {
        __m256d v = _mm256_loadu_pd(a + i);
        v = _mm256_and_pd(v, sign_mask);
        __m256d mask_lt = _mm256_cmp_pd(v, vmax, _CMP_LT_OQ);
        vmax = _mm256_blendv_pd(vmax, v, mask_lt);
        imax = _mm256_blendv_epi8(imax, vi, _mm256_castpd_si256(mask_lt));
        vi = _mm256_add_epi64(vi, vi_inc);
    }

    __attribute__((aligned(32))) double mv[4];
    __attribute__((aligned(32))) int64_t iv[4];
    _mm256_store_pd(mv, vmax);
    _mm256_store_si256((__m256i *)iv, imax);

    loc_t loc = {mv[0], iv[0]};
    for (int j = 1; j < 4; ++j) {
        if (mv[j] > loc.maxv || (mv[j] == loc.maxv && iv[j] < loc.index)) {
            loc.maxv = mv[j];
            loc.index = iv[j];
        }
    }

    for (; i < end; ++i) {
        double v = fabs(a[i]);
        if (v > loc.maxv || (v == loc.maxv && i < loc.index)) {
            loc.maxv = v;
            loc.index = i;
        }
    }
    return loc;
}

static inline loc_t argmax_range_avx2_strided(const double *a, int64_t start, int64_t end, int64_t inc) {
    double maxv0 = fabs(a[0]);
    __m256d vmax = _mm256_set1_pd(maxv0);
    __m256i imax = _mm256_set1_epi64x(0);
    __m256i vidx = _mm256_setr_epi64x(start * inc, (start + 1) * inc, (start + 2) * inc, (start + 3) * inc);
    __m256i vi = _mm256_setr_epi64x(start, start + 1, start + 2, start + 3);
    __m256i vidx_inc = _mm256_set1_epi64x(4 * inc);
    __m256i vi_inc = _mm256_set1_epi64x(4);
    __m256d sign_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));

    int64_t i;
    for (i = start; i + 3 < end; i += 4) {
        __m256d v = _mm256_i64gather_pd(a, vidx, 8);
        v = _mm256_and_pd(v, sign_mask);
        __m256d mask_lt = _mm256_cmp_pd(v, vmax, _CMP_LT_OQ);
        vmax = _mm256_blendv_pd(vmax, v, mask_lt);
        imax = _mm256_blendv_epi8(imax, vi, _mm256_castpd_si256(mask_lt));
        vidx = _mm256_add_epi64(vidx, vidx_inc);
        vi = _mm256_add_epi64(vi, vi_inc);
    }

    __attribute__((aligned(32))) double mv[4];
    __attribute__((aligned(32))) int64_t iv[4];
    _mm256_store_pd(mv, vmax);
    _mm256_store_si256((__m256i *)iv, imax);

    loc_t loc = {mv[0], iv[0]};
    for (int j = 1; j < 4; ++j) {
        if (mv[j] > loc.maxv || (mv[j] == loc.maxv && iv[j] < loc.index)) {
            loc.maxv = mv[j];
            loc.index = iv[j];
        }
    }

    int64_t k = i * inc;
    for (; i < end; ++i) {
        double v = fabs(a[k]);
        if (v > loc.maxv || (v == loc.maxv && i < loc.index)) {
            loc.maxv = v;
            loc.index = i;
        }
        k += inc;
    }
    return loc;
}

#endif

static inline loc_t argmax_range_scalar(const double *a, int64_t start, int64_t end, int64_t inc) {
    double maxv = fabs(a[0]);
    int64_t index = 0;
    int64_t k = start * inc;
    for (int64_t i = start; i < end; ++i) {
        double v = fabs(a[k]);
        if (v > maxv || (v == maxv && i < index)) {
            maxv = v;
            index = i;
        }
        k += inc;
    }
    return (loc_t){maxv, index};
}

void tsvc_2_s318_fp64(const double *restrict a, double *restrict result, const int64_t LEN_1D, const int64_t inc) {
    loc_t global = {fabs(a[0]), 0};

    if (LEN_1D <= 1) {
        result[0] = global.maxv + (double)global.index;
        return;
    }

    int max_threads = omp_get_max_threads();
    loc_t *locals = (loc_t *)__builtin_alloca((size_t)max_threads * sizeof(loc_t));
    for (int t = 0; t < max_threads; ++t) {
        locals[t] = global;
    }

    #pragma omp parallel if (LEN_1D > 4096)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int64_t total = LEN_1D - 1;
        int64_t chunk = (total + nthreads - 1) / nthreads;
        int64_t start = 1 + tid * chunk;
        int64_t end = start + chunk;
        if (end > LEN_1D) end = LEN_1D;

        loc_t local;
        if (inc == 1) {
#if defined(USE_AVX512)
            local = argmax_range_avx512_contig(a, start, end);
#elif defined(USE_AVX2)
            local = argmax_range_avx2_contig(a, start, end);
#else
            local = argmax_range_scalar(a, start, end, 1);
#endif
        } else {
#if defined(USE_AVX512)
            local = argmax_range_avx512_strided(a, start, end, inc);
#elif defined(USE_AVX2)
            local = argmax_range_avx2_strided(a, start, end, inc);
#else
            local = argmax_range_scalar(a, start, end, inc);
#endif
        }
        locals[tid] = local;
    }

    global = locals[0];
    for (int t = 1; t < max_threads; ++t) {
        global = loc_combine(global, locals[t]);
    }

    result[0] = global.maxv + (double)global.index;
}
