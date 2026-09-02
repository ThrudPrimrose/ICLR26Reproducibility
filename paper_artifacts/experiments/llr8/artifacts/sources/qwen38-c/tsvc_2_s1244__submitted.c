/* TSVC_2 s1244 optimized.
 *
 * Reference:
 *   for i in 0..LEN_1D-2:
 *       a[i] = b[i] + c[i]*c[i] + b[i]*b[i] + c[i]
 *       d[i] = a[i] + a[i+1]          (a[i+1] is still the OLD value)
 *
 * With t[i] = ((b[i]+c[i]^2)+b[i]^2)+c[i]:  a[i]=t[i] (i<LEN-1), d[i]=t[i]+a_old[i+1].
 *
 * Threading is safe in two phases (barrier between):
 *   phase 1: d[i] = f(b,c) + a[i+1]   -- reads a (old), writes d only
 *   phase 2: a[i] = f(b,c)            -- writes a, reads b,c only
 * Hand-vectorized AVX-512 (8 dbl/vec), up to 16 elements per chunk.
 * No FMA (plain mul/add) => bit-exact vs the no-FMA NumPy reference.
 */
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <omp.h>

static inline __m512d f_bc(__m512d b, __m512d c) {
    return _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(b, _mm512_mul_pd(c, c)),
                                       _mm512_mul_pd(b, b)),
                         c);
}

/* fused: a[i..] = t, d[i..] = t + a[i+1..]; chunk of cnt<=16 elements at i */
static inline void fused_chunk(double* a, double* b, double* c, double* d,
                               int64_t i, int64_t cnt) {
    if (cnt >= 8) {
        __m512d bo = _mm512_loadu_pd(b + i);
        __m512d co = _mm512_loadu_pd(c + i);
        __m512d ap = _mm512_loadu_pd(a + i + 1);
        __m512d t  = f_bc(bo, co);
        _mm512_storeu_pd(a + i, t);
        _mm512_storeu_pd(d + i, _mm512_add_pd(t, ap));
        i += 8; cnt -= 8;
    }
    if (cnt > 0) {
        const __mmask8 m = (__mmask8)((1u << cnt) - 1u);
        __m512d bo = _mm512_maskz_loadu_pd(m, b + i);
        __m512d co = _mm512_maskz_loadu_pd(m, c + i);
        __m512d ap = _mm512_maskz_loadu_pd(m, a + i + 1);
        __m512d t  = f_bc(bo, co);
        _mm512_mask_storeu_pd(a + i, m, t);
        _mm512_mask_storeu_pd(d + i, m, _mm512_add_pd(t, ap));
    }
}

/* phase 1 only: d[i..] = f(b,c) + a[i+1..]; chunk of cnt<=16 at i */
static inline void p1_chunk(double* a, double* b, double* c, double* d,
                            int64_t i, int64_t cnt) {
    if (cnt >= 8) {
        __m512d bo = _mm512_loadu_pd(b + i);
        __m512d co = _mm512_loadu_pd(c + i);
        __m512d ap = _mm512_loadu_pd(a + i + 1);
        __m512d t  = f_bc(bo, co);
        _mm512_storeu_pd(d + i, _mm512_add_pd(t, ap));
        i += 8; cnt -= 8;
    }
    if (cnt > 0) {
        const __mmask8 m = (__mmask8)((1u << cnt) - 1u);
        __m512d bo = _mm512_maskz_loadu_pd(m, b + i);
        __m512d co = _mm512_maskz_loadu_pd(m, c + i);
        __m512d ap = _mm512_maskz_loadu_pd(m, a + i + 1);
        __m512d t  = f_bc(bo, co);
        _mm512_mask_storeu_pd(d + i, m, _mm512_add_pd(t, ap));
    }
}

/* phase 2 only: a[i..] = f(b,c); chunk of cnt<=16 at i */
static inline void p2_chunk(double* a, double* b, double* c,
                            int64_t i, int64_t cnt) {
    if (cnt >= 8) {
        __m512d bo = _mm512_loadu_pd(b + i);
        __m512d co = _mm512_loadu_pd(c + i);
        _mm512_storeu_pd(a + i, f_bc(bo, co));
        i += 8; cnt -= 8;
    }
    if (cnt > 0) {
        const __mmask8 m = (__mmask8)((1u << cnt) - 1u);
        __m512d bo = _mm512_maskz_loadu_pd(m, b + i);
        __m512d co = _mm512_maskz_loadu_pd(m, c + i);
        _mm512_mask_storeu_pd(a + i, m, f_bc(bo, co));
    }
}

void tsvc_2_s1244_fp64(double* a, double* b, double* c, double* d,
                       int64_t LEN_1D, uint8_t* workspace, int64_t workspace_bytes) {
    (void)workspace; (void)workspace_bytes;
    const int64_t n = LEN_1D - 1;
    if (n <= 0) return;

    if (n >= (1 << 18) && omp_get_max_threads() > 1) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < n; i += 16)
                p1_chunk(a, b, c, d, i, n - i < 16 ? n - i : 16);
            #pragma omp barrier
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < n; i += 16)
                p2_chunk(a, b, c, i, n - i < 16 ? n - i : 16);
        }
        return;
    }

    for (int64_t i = 0; i < n; i += 16)
        fused_chunk(a, b, c, d, i, n - i < 16 ? n - i : 16);
}
