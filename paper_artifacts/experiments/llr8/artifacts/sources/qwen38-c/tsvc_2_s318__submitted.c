#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

/* TSVC s318: result[0] = max(|a[i*inc]| for i in [0,LEN_1D)) + (double)first_index,
   where first_index is the FIRST i attaining the maximum (strict-greater semantics). */

#define MAXT 4096
static double  s_pv[MAXT];
static int64_t s_pp[MAXT];

static void combine_partial(double *out_v, int64_t *out_p, double v, int64_t p)
{
    if (v > *out_v || (v == *out_v && p < *out_p)) { *out_v = v; *out_p = p; }
}

static void scalar_range(const double *a, int64_t i0, int64_t i1, int64_t inc,
                         double *bv, int64_t *bp)
{
    for (int64_t i = i0; i < i1; i++) {
        double v = fabs(a[i * inc]);
        if (v > *bv) { *bv = v; *bp = i; }
    }
}

void tsvc_2_s318_fp64(double *a, double *result, int64_t LEN_1D, int64_t inc,
                      unsigned char *workspace, int64_t workspace_bytes)
{
    (void)workspace; (void)workspace_bytes;
    if (LEN_1D <= 0) { result[0] = 0.0; return; }

    if (inc != 1 || LEN_1D < (1 << 20)) {
        /* faithful scalar / strided path (also the small-N path) */
        double gmax = -INFINITY;
        int64_t gidx = 0;
        if (LEN_1D <= 4096) {
            for (int64_t i = 0; i < LEN_1D; i++) {
                double v = fabs(a[i * inc]);
                if (v > gmax) { gmax = v; gidx = i; }
            }
        } else {
            int nts_used = 1;
            #pragma omp parallel
            {
                int nts = omp_get_num_threads();
                int tid = omp_get_thread_num();
                int64_t i0 = (LEN_1D / nts) * tid;
                int64_t i1 = (tid == nts - 1) ? LEN_1D : (LEN_1D / nts) * (tid + 1);
                double bv = -INFINITY; int64_t bp = i0;
                scalar_range(a, i0, i1, inc, &bv, &bp);
                s_pv[tid] = bv; s_pp[tid] = bp;
                #pragma omp critical
                nts_used = nts;
            }
            for (int t = 0; t < nts_used; t++)
                if (s_pv[t] > -INFINITY) combine_partial(&gmax, &gidx, s_pv[t], s_pp[t]);
        }
        result[0] = gmax + (double)gidx;
        return;
    }

    /* inc == 1, large: contiguous, vectorized, first-occurrence tracked per lane */
    double gmax = -INFINITY;
    int64_t gidx = 0;
    int nts_used = 1;
    #pragma omp parallel
    {
        int nts = omp_get_num_threads();
        int tid = omp_get_thread_num();
        if (nts > MAXT) nts = MAXT; /* won't happen; guard anyway */
        int64_t c0 = (LEN_1D / (8 * nts)) * 8 * tid;
        int64_t c1 = (tid == nts - 1) ? LEN_1D : (LEN_1D / (8 * nts)) * 8 * (tid + 1);

        double bv = -INFINITY;
        int64_t bp = c0;
        if (c0 < c1) {
            __m512d vmax = _mm512_set1_pd(-INFINITY);
            __m512d vpos = _mm512_setzero_pd();
            const __m512d sgnmask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7fffffffffffffffLL));
            const __m512d pos8 = _mm512_setr_pd(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
            const double *p = a + c0;
            const double *pe = a + c1;
            for (; p + 8 <= pe; p += 8) {
                __m512d v = _mm512_loadu_pd(p);
                __m512d av = _mm512_and_pd(v, sgnmask);
                __mmask8 m = _mm512_cmp_pd_mask(av, vmax, _CMP_GT_OQ);
                vmax = _mm512_mask_blend_pd(m, vmax, av);
                vpos = _mm512_mask_blend_pd(m, vpos,
                                            _mm512_add_pd(_mm512_set1_pd((double)(p - a)), pos8));
            }
            for (; p < pe; p++) {
                double v = fabs(*p);
                if (v > bv) { bv = v; bp = (int64_t)(p - a); }
            }
            double vv[8], vp[8];
            _mm512_storeu_pd(vv, vmax);
            _mm512_storeu_pd(vp, vpos);
            for (int l = 0; l < 8; l++)
                if (vv[l] > bv || (vv[l] == bv && vp[l] < (double)bp)) { bv = vv[l]; bp = (int64_t)vp[l]; }
        }
        s_pv[tid] = bv; s_pp[tid] = bp;
        #pragma omp critical
        nts_used = nts;
    }
    for (int t = 0; t < nts_used; t++)
        if (s_pv[t] > -INFINITY) combine_partial(&gmax, &gidx, s_pv[t], s_pp[t]);
    result[0] = gmax + (double)gidx;
}
