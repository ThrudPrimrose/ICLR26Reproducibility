/* TSVC tsvc_2 s252, fp64 single-invocation variant -- optimized.
 *
 * Reference semantics:
 *   t = 0.0
 *   for i: s = b[i]*c[i]; a[i] = s + t; t = s
 * i.e.  a[i] = s[i] + s[i-1]  with s[i] = b[i]*c[i] and s[-1] = +0.0.
 * Every product and addition below is the same one rounding the reference
 * performs, so results are bit-identical to the sequential loop, while the
 * loop-carried dependency is removed: a thread owning elements [i0,i1) needs
 * only t = b[i0-1]*c[i0-1] (or +0.0 for i0 == 0), which it computes itself.
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s252_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {

    if (LEN_1D <= 0) return;

    /* Small input: serial is faster than paying team-setup cost. */
    if (LEN_1D <= 2048) {
        double t = 0.0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            double s = b[i] * c[i];
            a[i] = s + t;
            t = s;
        }
        return;
    }

    /* Per-lane shift of s right by one element: double lane i takes s[i-1]
     * for i >= 1; lane 0 is overwritten with t below. The double-lane index
     * sits in the LOW epi32 lane of each 64-bit slot. (8 double lanes per zmm.) */
    __m512i idx = _mm512_setr_epi32(0, 0,  0, 0,  1, 0,  2, 0,  3, 0,  4, 0,  5, 0,  6, 0);

    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < LEN_1D; i0 += 8) {
        int64_t i1 = i0 + 8;
        if (i1 > LEN_1D) i1 = LEN_1D;
        double t = (i0 > 0) ? b[i0 - 1] * c[i0 - 1] : 0.0;
        if (i1 - i0 < 8) {
            for (int64_t i = i0; i < i1; ++i) {
                double s = b[i] * c[i];
                a[i] = s + t;
                t = s;
            }
            continue;
        }
        __m512d s = _mm512_mul_pd(_mm512_loadu_pd(b + i0), _mm512_loadu_pd(c + i0));
        __m512d shifted = _mm512_permutexvar_pd(idx, s);                    /* lanes 1..7 = s[0..6] */
        shifted = _mm512_mask_blend_pd((__mmask8)1, shifted, _mm512_set1_pd(t)); /* lane 0 = t */
        _mm512_storeu_pd(a + i0, _mm512_add_pd(s, shifted));
    }
}
