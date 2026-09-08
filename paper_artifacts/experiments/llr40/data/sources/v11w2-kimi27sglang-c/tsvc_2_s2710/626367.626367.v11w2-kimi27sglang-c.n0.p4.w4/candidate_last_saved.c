#define NTH 8

#include <stdint.h>
#include <immintrin.h>

static inline void process8(const __m512d vone, const int len_gt_10, const int x_pos,
                            double *restrict a, double *restrict b, double *restrict c,
                            const double *restrict d, const double *restrict e, int64_t i) {
    __m512d ai = _mm512_loadu_pd(&a[i]);
    __m512d bi = _mm512_loadu_pd(&b[i]);
    __m512d ci = _mm512_loadu_pd(&c[i]);
    __m512d di = _mm512_loadu_pd(&d[i]);
    __m512d ei = _mm512_loadu_pd(&e[i]);

    __mmask8 m = _mm512_cmp_pd_mask(ai, bi, _CMP_GT_OQ);
    __m512d di2 = _mm512_mul_pd(di, di);
    __m512d ei2 = _mm512_mul_pd(ei, ei);

    __m512d a_new = _mm512_fmadd_pd(bi, di, ai);
    __m512d b_new = _mm512_add_pd(ai, ei2);

    __m512d c_true, c_false;
    if (len_gt_10) {
        c_true = _mm512_add_pd(ci, di2);
    } else {
        c_true = _mm512_fmadd_pd(di, ei, vone);
    }
    if (x_pos) {
        c_false = _mm512_add_pd(ai, di2);
    } else {
        c_false = _mm512_add_pd(ci, ei2);
    }

    _mm512_mask_storeu_pd(&a[i], m, a_new);
    _mm512_mask_storeu_pd(&b[i], (__mmask8)(~m), b_new);
    __m512d c_out = _mm512_mask_blend_pd(m, c_false, c_true);
    _mm512_storeu_pd(&c[i], c_out);
}

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b, double *restrict c,
                       const double *restrict d, const double *restrict e,
                       const double *restrict x, const int64_t LEN_1D) {
    const int len_gt_10 = LEN_1D > 10;
    const int x_pos = x[0] > 0.0;
    const int64_t nvec = LEN_1D & ~7;
    int64_t i = 0;
    if (nvec > 0) {
        const __m512d vone = _mm512_set1_pd(1.0);
        if (LEN_1D > 4096) {
            const int64_t nvec2 = nvec & ~15;
            #pragma omp parallel for num_threads(NTH) schedule(static)
            for (i = 0; i < nvec2; i += 16) {
                process8(vone, len_gt_10, x_pos, a, b, c, d, e, i);
                process8(vone, len_gt_10, x_pos, a, b, c, d, e, i + 8);
            }
            for (i = nvec2; i < nvec; i += 8) {
                process8(vone, len_gt_10, x_pos, a, b, c, d, e, i);
            }
        } else {
            for (i = 0; i < nvec; i += 8) {
                process8(vone, len_gt_10, x_pos, a, b, c, d, e, i);
            }
        }
        i = nvec;
    }
    for (; i < LEN_1D; ++i) {
        if (a[i] > b[i]) {
            a[i] += b[i] * d[i];
            if (len_gt_10)
                c[i] += d[i] * d[i];
            else
                c[i] = d[i] * e[i] + 1.0;
        } else {
            b[i] = a[i] + e[i] * e[i];
            if (x_pos)
                c[i] = a[i] + d[i] * d[i];
            else
                c[i] += e[i] * e[i];
        }
    }
}
