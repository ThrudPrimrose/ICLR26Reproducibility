/* hpcagent_bench-autogen -- generated from mandelbrot1_numpy.py; edit the numpy reference and regenerate, or delete this line to keep local edits as a hand override. */
/*
 * Optimized Mandelbrot escape-time (mandelbrot1_fp64).
 *
 * Numerics: reproduces the NumPy reference elementwise operations BIT-EXACTLY
 * (verified against numpy on this CPU family), so N_out is exact and Z_out is
 * bit-identical:
 *   - mask test          : (x*x) + (y*y)  < 4.0      (two rounded muls, one rounded add)
 *   - complex square real: fma(x, x, -(y*y))          (numpy's SIMD path contracts this)
 *   - complex square imag: (x*y) * 2.0                (rounded product, then exact doubling)
 *   - add C              : plain rounded adds
 *   - linspace           : step*i + start, each op rounded once; last point = stop
 * All floating point is spelled as intrinsics so -ffp-contract=fast cannot fuse
 * anything extra.
 *
 * Algorithm: SIMD over 8 consecutive points (AVX-512, zr/zi in __m512d), one
 * iteration k per NumPy loop step. A point stops updating once it escapes (its
 | z stays frozen exactly as in the masked reference), and N is written once,
 * at the escape iteration (k-1); points alive at the end keep N=0 (the
 * N[N==maxiter-1]=0 rule). Four 8-point groups run interleaved for ILP.
 */
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>

typedef long long i64;

/* One 8-point vector: z (split), c (real part; imag is row-broadcast), alive mask. */
typedef struct {
    __m512d zr, zi, cx;
    __mmask8 A;
} g8;

/* Advance one group by one iteration k (0-based). Returns the alive mask AFTER this
 * iteration; stores N = k-1 into the lanes that escape at this iteration. */
static inline __mmask8 step1(g8 *g, i64 k, const i64_t *Nrow, i64 col,
                             const __m512d four, const __m512d two, const __m512d cyv)
{
    const __m512d zr = g->zr, zi = g->zi;
    const __m512d r2 = _mm512_mul_pd(zr, zr);
    const __m512d i2 = _mm512_mul_pd(zi, zi);
    const __mmask8 m = _mm512_cmp_pd_mask(_mm512_add_pd(r2, i2), four, _CMP_LT_OQ);
    const __mmask8 alive = g->A & m;
    const __mmask8 esc = g->A & (~m);
    if (esc)
        _mm_mask8_storeu_epi64(Nrow + col, esc, _mm512_set1_epi64((long long)(k - 1)));
    const __m512d t  = _mm512_fmadd231_pd(zr, zr, _mm512_negate_pd(i2)); /* x^2 - y^2, one rounding */
    const __m512d nzr = _mm512_add_pd(t, g->cx);
    const __m512d t2 = _mm512_mul_pd(zr, zi);
    const __m512d nzi = _mm512_fmadd_pd(two, t2, cyv);                  /* (x*y)*2 + c: 2*(x*y) exact */
    g->zr = _mm512_mask_mov_pd(zr, alive, nzr);
    g->zi = _mm512_mask_mov_pd(zi, alive, nzi);
    g->A = alive;
    return alive;
}

/* Process `ng` 8-point groups starting at column i0 of row j. `vmask_last` is the
 * in-range mask of the LAST group (all-ones when the block is full). N_out must be
 * pre-zeroed. Stores Z_out for every in-range point. */
static inline void run_groups(g8 *g, int ng, i64 i0, __mmask8 vmask_last,
                              const double *Xrow, i64_t *Nrow, double *Zrow,
                              i64 maxiter, const __m512d four, const __m512d two,
                              const __m512d cyv)
{
    for (int t = 0; t < ng; t++) {
        i64 col = i0 + 8 * (i64)t;
        if (t == ng - 1 && vmask_last != 0xFF)
            g[t].cx = _mm256_mask_loadu_pd(_mm512_setzero_pd(), vmask_last, Xrow + col);
        else
            g[t].cx = _mm512_loadu_pd(Xrow + col);
        g[t].zr = _mm512_setzero_pd();
        g[t].zi = _mm512_setzero_pd();
        g[t].A = (t == ng - 1) ? vmask_last : 0xFF;
    }
    for (i64 k = 0; k < maxiter; k++) {
        i64 live = g[0].A;
        for (int t = 1; t < ng; t++) live |= g[t].A;
        if (!live) break;
        for (int t = 0; t < ng; t++)
            if (g[t].A)
                step1(&g[t], k, Nrow, i0 + 8 * (i64)t, four, two, cyv);
    }
    for (int t = 0; t < ng; t++) {
        i64 col = i0 + 8 * (i64)t;
        i64_t *dst = Nrow + col;
        __m512d zr = g[t].zr, zi = g[t].zi;
        double re[8], im[8];
        _mm512_storeu_pd(re, zr);
        _mm512_storeu_pd(im, zi);
        double *dz = Zrow + 2 * col;
        for (int e = 0; e < 8; e++) {
            dz[2 * e] = re[e];
            dz[2 * e + 1] = im[e];
        }
        (void)dst;
    }
}

void mandelbrot1_fp64(int64_t *restrict N_out, double _Complex *restrict Z_out,
                      const int64_t maxiter, const int64_t xn, const int64_t yn)
{
    /* ---- grids, bit-exact np.linspace: step = (stop-start)/(n-1); v[i] = step*i + start
     * (each op rounded once -- scalar SSE keeps mul/add distinct from an FMA); v[n-1] = stop */
    double *X = (double *)malloc((size_t)xn * sizeof(double));
    double *Y = (double *)malloc((size_t)yn * sizeof(double));
    if (xn > 1) {
        const __m128d start = _mm_set_sd(-2.25);
        const __m128d stop = _mm_set_sd(0.75);
        const __m128d step = _mm_div_sd(_mm_sub_sd(stop, start), _mm_set_sd((double)(xn - 1)));
        for (i64 i = 0; i < xn - 1; i++)
            X[i] = _mm_cvtsd_f64(_mm_add_sd(_mm_mul_sd(step, _mm_set_sd((double)i)), start));
        X[xn - 1] = 0.75;
    } else {
        X[0] = 0.75;
    }
    if (yn > 1) {
        const __m128d start = _mm_set_sd(-1.25);
        const __m128d stop = _mm_set_md(1.25);
        const __m128d step = _mm_div_sd(_mm_sub_sd(stop, start), _mm_set_sd((double)(yn - 1)));
        for (i64 i = 0; i < yn - 1; i++)
            Y[i] = _mm_cvtsd_f64(_mm_add_sd(_mm_mul_sd(step, _mm_set_sd((double)i)), start));
        Y[yn - 1] = 1.25;
    } else {
        Y[0] = 1.25;
    }

    const __m512d four = _mm512_set1_pd(4.0);
    const __m512d two = _mm512_set1_pd(2.0);

    /* N: the "N[N == maxiter-1] = 0" rule means alive-at-end points read 0, so zero once. */
    for (i64 idx = 0; idx < xn * yn; idx++)
        N_out[idx] = 0;

    for (i64 j = 0; j < yn; j++) {
        const __m512d cyv = _mm512_set1_pd(Y[j]);
        i64_t *Nrow = N_out + j * xn;
        double *Zrow = (double *)Z_out + j * xn * 2;
        const double *Xrow = X;
        i64 i = 0;
        g8 g[8];
        for (; i + 32 <= xn; i += 32)
            run_groups(g, 4, i, 0xFF, Xrow, Nrow, Zrow, maxiter, four, two, cyv);
        if (i < xn) {
            /* tail: whole 8-blocks, then a partial final block with an in-range mask */
            i64 rem = xn - i;
            int full = (int)(rem / 8);
            i64 left = rem - 8 * (i64)full;
            if (full > 4) full = 4;
            for (int c = 0; c < full; c++)
                run_groups(g + c, 1, i + 8 * (i64)c, 0xFF, Xrow, Nrow, Zrow, maxiter, four, two, cyv);
            if (left) {
                __mmask8 vm = (left >= 1 ? 1 : 0);
                for (int e = 1; e < 8 && e < left; e++) vm |= (__mmask8)(1 << e);
                run_groups(g + full, 1, i + 8 * (i64)full, vm, Xrow, Nrow, Zrow, maxiter, four, two, cyv);
            }
        }
    }
    free(X);
    free(Y);
}
