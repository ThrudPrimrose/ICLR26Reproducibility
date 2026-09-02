#include <stdint.h>
#include <omp.h>
#include <math.h>
#include <immintrin.h>

typedef struct {
    double v;
    int64_t i;
    int64_t j;
} partial_t;

#pragma omp declare reduction(maxloc : partial_t : \
    omp_out = (omp_in.v > omp_out.v) ? omp_in : \
              (omp_in.v < omp_out.v) ? omp_out : \
              ((omp_in.i < omp_out.i) || \
               (omp_in.i == omp_out.i && omp_in.j < omp_out.j)) ? omp_in : omp_out) \
    initializer(omp_priv = {-HUGE_VAL, -1, -1})

static inline void argmax_row(const double *restrict row, int64_t n,
                              double *restrict out_v, int64_t *restrict out_j) {
    __m512d vmax = _mm512_set1_pd(-HUGE_VAL);
    __m512i jbase = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    const __m512i jinc = _mm512_set1_epi64(8);
    __m512i jmax = _mm512_set1_epi64(0);

    int64_t j = 0;
    for (; j + 8 <= n; j += 8) {
        __m512d v = _mm512_loadu_pd(row + j);
        __mmask8 gt = _mm512_cmp_pd_mask(v, vmax, _CMP_GT_OQ);
        vmax = _mm512_mask_blend_pd(gt, vmax, v);
        jmax = _mm512_mask_blend_epi64(gt, jmax, jbase);
        jbase = _mm512_add_epi64(jbase, jinc);
    }

    double vals[8];
    int64_t idxs[8];
    _mm512_storeu_pd(vals, vmax);
    _mm512_storeu_si512((__m512i *)idxs, jmax);

    double best_v = -HUGE_VAL;
    int64_t best_j = 0;
    for (int l = 0; l < 8; ++l) {
        double v = vals[l];
        if (v > best_v) {
            best_v = v;
            best_j = idxs[l];
        }
    }

    for (; j < n; ++j) {
        double v = row[j];
        if (v > best_v) {
            best_v = v;
            best_j = j;
        }
    }

    *out_v = best_v;
    *out_j = best_j;
}

void tsvc_2_s3110_fp64(double *restrict aa, double *restrict bb,
                       int64_t LEN_2D, uint8_t *restrict workspace,
                       int64_t workspace_bytes) {
    (void)workspace;
    (void)workspace_bytes;

    partial_t best = {-HUGE_VAL, -1, -1};

    #pragma omp parallel for reduction(maxloc:best) schedule(static)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        double rowmax;
        int64_t jmax;
        argmax_row(aa + i * LEN_2D, LEN_2D, &rowmax, &jmax);
        if (rowmax > best.v) {
            best.v = rowmax;
            best.i = i;
            best.j = jmax;
        }
    }

    bb[0] = best.v + (double)best.i + (double)best.j;
}
