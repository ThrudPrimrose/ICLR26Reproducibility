#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#ifndef PFDIST
#define PFDIST 64
#endif

#define VEC_BODY(ii, gt_macro, c_out_expr) do { \
    __m512d va = _mm512_loadu_pd(a + (ii)); \
    __m512d vb = _mm512_loadu_pd(b + (ii)); \
    __m512d vc = _mm512_loadu_pd(c + (ii)); \
    __m512d vd = _mm512_loadu_pd(d + (ii)); \
    __m512d ve = _mm512_loadu_pd(e + (ii)); \
    __mmask8 gt = _mm512_cmp_pd_mask(va, vb, _CMP_GT_OQ); \
    __m512d vd2 = _mm512_mul_pd(vd, vd); \
    __m512d ve2 = _mm512_mul_pd(ve, ve); \
    __m512d a_out = _mm512_mask3_fmadd_pd(vb, vd, va, gt); \
    __m512d b_out = _mm512_mask_add_pd(vb, (__mmask8)((~gt) & 0xFF), va, ve2); \
    __m512d c_out = (c_out_expr); \
    _mm512_storeu_pd(a + (ii), a_out); \
    _mm512_storeu_pd(b + (ii), b_out); \
    _mm512_storeu_pd(c + (ii), c_out); \
} while(0)

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    if (n <= 0) return;

    const int len_gt10 = n > 10;
    const int xpos = x[0] > 0.0;

    int threads = 1;
    int maxt = omp_get_max_threads();
    if (n > 4096) {
        threads = maxt;
        if (threads > 48) threads = 96;
    }

    int64_t i = 0;

    if (n >= 16 && len_gt10) {
        const int64_t n16 = n & ~((int64_t)15);
        if (xpos) {
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (int64_t ii = 0; ii < n16; ii += 16) {
                _mm_prefetch((const char*)(a + ii + PFDIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + ii + PFDIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(c + ii + PFDIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(d + ii + PFDIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(e + ii + PFDIST), _MM_HINT_T0);
                VEC_BODY(ii, gt, _mm512_add_pd(_mm512_mask_blend_pd(gt, va, vc), vd2));
                VEC_BODY(ii + 8, gt, _mm512_add_pd(_mm512_mask_blend_pd(gt, va, vc), vd2));
            }
        } else {
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (int64_t ii = 0; ii < n16; ii += 16) {
                _mm_prefetch((const char*)(a + ii + PFDIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(b + ii + PFDIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(c + ii + PFDIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(d + ii + PFDIST), _MM_HINT_T0);
                _mm_prefetch((const char*)(e + ii + PFDIST), _MM_HINT_T0);
                VEC_BODY(ii, gt, _mm512_add_pd(vc, _mm512_mask_blend_pd(gt, ve2, vd2)));
                VEC_BODY(ii + 8, gt, _mm512_add_pd(vc, _mm512_mask_blend_pd(gt, ve2, vd2)));
            }
        }
        i = n16;
    }

    for (; i < n; ++i) {
        double ai = a[i];
        double bi = b[i];
        double di = d[i];
        double ei = e[i];
        if (ai > bi) {
            a[i] = ai + bi * di;
            if (len_gt10) {
                c[i] = c[i] + di * di;
            } else {
                c[i] = di * ei + 1.0;
            }
        } else {
            b[i] = ai + ei * ei;
            if (xpos) {
                c[i] = ai + di * di;
            } else {
                c[i] = c[i] + ei * ei;
            }
        }
    }
}
