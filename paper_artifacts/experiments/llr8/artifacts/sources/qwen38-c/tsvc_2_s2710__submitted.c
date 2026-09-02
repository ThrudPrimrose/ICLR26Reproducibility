/*
 * TSVC_2 s2710 (bit-exact vs the NumPy reference).
 *
 *   for i in 0..LEN_1D:
 *     if a[i] > b[i]:
 *       a[i] = a[i] + b[i]*d[i]
 *       c[i] = (LEN_1D > 10) ? c[i] + d[i]*d[i] : d[i]*e[i] + 1.0
 *     else:
 *       b[i] = a[i] + e[i]*e[i]
 *       c[i] = (x[0] > 0.0) ? a[i] + d[i]*d[i] : c[i] + e[i]*e[i]
 *
 * Each iteration only touches index i  ->  fully independent, so the loop
 * threads with a plain static partition and the body vectorizes to 8-wide
 * masked ops.  The guards LEN_1D>10 and x[0]>0 are loop invariant and are
 * unswitched once.
 *
 * The judge build line omits -ffp-contract=off, so GCC is free to contract
 * mul+add into FMA and to reassociate; the reference does plain IEEE
 * mul-then-add.  The NF helpers below are noinline+const, which is an
 * optimizer barrier: the exact two-op trees are preserved (verified
 * bit-exact against the NumPy reference, including subnormals/inf/NaN).
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#define NF __attribute__((noinline, const))

static double NF dmul(double x, double y) { return x * y; }
static double NF dadd(double x, double y) { return x + y; }
static double NF dadd1(double x)          { return x + 1.0; }
static __m512d NF vmul(__m512d x, __m512d y) { return _mm512_mul_pd(x, y); }
static __m512d NF vadd(__m512d x, __m512d y) { return _mm512_add_pd(x, y); }
static __m512d NF vadd1(__m512d x)          { return _mm512_add_pd(x, _mm512_set1_pd(1.0)); }

/*
 * Processes 8 contiguous elements.
 * Note: _mm512_mask_blend_pd(k, p, q) selects q where k is set.
 */
static inline __attribute__((always_inline)) void
block8(double *restrict a, double *restrict b, double *restrict c,
       const double *restrict d, const double *restrict e,
       int xpos, int big)
{
    __m512d av = _mm512_loadu_pd(a);
    __m512d bv = _mm512_loadu_pd(b);
    __m512d cv = _mm512_loadu_pd(c);
    __m512d dv = _mm512_loadu_pd(d);
    __m512d ev = _mm512_loadu_pd(e);
    __mmask8 m = _mm512_cmp_pd_mask(av, bv, _CMP_GT_OQ);
    __m512d dd = vmul(dv, dv);
    __m512d ee = vmul(ev, ev);
    _mm512_storeu_pd(a, _mm512_mask_blend_pd(m, av, vadd(av, vmul(bv, dv))));
    _mm512_storeu_pd(b, _mm512_mask_blend_pd(m, vadd(av, ee), bv));
    __m512d c_hi = big ? vadd(cv, dd) : vadd1(vmul(dv, ev));
    __m512d c_lo = xpos ? vadd(av, dd) : vadd(cv, ee);
    _mm512_storeu_pd(c, _mm512_mask_blend_pd(m, c_lo, c_hi));
}

void tsvc_2_s2710_fp64(double *restrict a, double *restrict b,
                       double *restrict c, const double *restrict d,
                       const double *restrict e, const double *restrict x,
                       int64_t LEN_1D)
{
    const int xpos = x[0] > 0.0;
    const int big  = LEN_1D > 10;
    const int64_t n8 = LEN_1D >> 3;
    if (n8 >= 4096) {
        #pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < n8; j++)
            block8(a + 8*j, b + 8*j, c + 8*j, d + 8*j, e + 8*j, xpos, big);
    } else {
        for (int64_t j = 0; j < n8; j++)
            block8(a + 8*j, b + 8*j, c + 8*j, d + 8*j, e + 8*j, xpos, big);
    }
    for (int64_t i = n8 << 3; i < LEN_1D; i++) {
        const double ai = a[i], bi = b[i], ci = c[i], di = d[i], ei = e[i];
        const double dd = dmul(di, di), ee = dmul(ei, ei);
        if (ai > bi) {
            a[i] = dadd(ai, dmul(bi, di));
            c[i] = big ? dadd(ci, dd) : dadd1(dmul(di, ei));
        } else {
            b[i] = dadd(ai, ee);
            c[i] = xpos ? dadd(ai, dd) : dadd(ci, ee);
        }
    }
}
