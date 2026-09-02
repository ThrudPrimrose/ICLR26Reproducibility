#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

void tsvc_2_s316_fp64(double *a, double *result, int64_t LEN_1D,
                      uint8_t *workspace, int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    if (LEN_1D <= 0) return;
    const int nt = omp_get_max_threads();
    const int64_t chunk = (LEN_1D + nt - 1) / nt;
    const double INF = __builtin_huge_val();
    double x = INF;
    #pragma omp parallel for reduction(min:x) schedule(static)
    for (int64_t t = 0; t < nt; ++t) {
        const int64_t lo = t * chunk;
        int64_t hi = lo + chunk;
        if (hi > LEN_1D) hi = LEN_1D;
        double p = INF;
        int64_t i = lo;
#ifdef __AVX512F__
        __m512d vx = _mm512_set1_pd(INF);
        const int64_t end = hi - 7;
        for (; i < end; i += 8) {
            __m512d va = _mm512_loadu_pd(a + i);
            vx = _mm512_mask_blend_pd(_mm512_cmplt_pd_mask(va, vx), vx, va);
        }
        p = _mm512_reduce_min_pd(vx);
#endif
        for (; i < hi; ++i) {
            if (a[i] < p) p = a[i];
        }
        if (p < x) x = p;
    }
    *result = x;
}
