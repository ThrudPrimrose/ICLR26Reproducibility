/* TSVC s3111: conditional sum reduction (only a[i] > 0.0 contributes).
 *
 * Bit-deterministic across runs AND thread counts:
 *  - fixed number of chunks (a function of LEN_1D only);
 *  - each chunk sum is a pure function of its elements
 *    (explicit AVX-512, fixed unroll, fixed lane-combine order);
 *  - final combine is a single-threaded sequential sum over the
 *    chunk sums in index order.
 *
 * (x > 0) ? x : 0.0 == max(x, 0.0) for every double (matches the
 * branch even for NaN: NaN > 0 is false and vmax(NaN, 0) == 0).
 *
 * The zero vector is materialized with a full 512-bit vpxor written as
 * inline asm: this toolchain (GCC 16) materializes its own constant
 * zero as a 128-bit `vpxor %xmm` whose upper 384 bits are left
 * undefined, then consumes that register at full 512-bit width
 * (vmaxpd operand and the 8 accumulator inits) -- silently corrupting
 * the result whenever the entry ZMM state is dirty.
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#define MAX_CHUNKS 256

/* Zero vector loaded from real memory. The volatile pointer defeats constant
 * folding: this toolchain (GCC 16) materializes its own compile-time zero
 * vectors as 128-bit `vpxor %xmm` whose upper 384 zmm bits stay undefined,
 * and then consumes the register at full 512-bit width (vmaxpd operand,
 * accumulator inits) -- silently corrupting the sum whenever the entry ZMM
 * state is dirty. A genuine load of this all-zero array is immune. */
static const double kzero[8] __attribute__((aligned(64))) =
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static const double *const volatile kzero_ptr = kzero;

static __attribute__((noinline, optimize("no-tree-vectorize")))
double tail_sum(const double *restrict p, int64_t n)
{
    double t = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double x = p[i];
        if (x > 0.0) t += x;
    }
    return t;
}

static __attribute__((noinline)) double sum_range(const double *restrict a, int64_t i0, int64_t i1)
{
    const double *p = a + i0;
    int64_t n = i1 - i0;
    __m512d zero = _mm512_load_pd(kzero_ptr);
    __m512d s0 = zero, s1 = zero, s2 = zero, s3 = zero;
    __m512d s4 = zero, s5 = zero, s6 = zero, s7 = zero;
    int64_t i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512d v;
        v = _mm512_loadu_pd(p + i);       s0 = _mm512_add_pd(s0, _mm512_max_pd(v, zero));
        v = _mm512_loadu_pd(p + i + 8);   s1 = _mm512_add_pd(s1, _mm512_max_pd(v, zero));
        v = _mm512_loadu_pd(p + i + 16);  s2 = _mm512_add_pd(s2, _mm512_max_pd(v, zero));
        v = _mm512_loadu_pd(p + i + 24);  s3 = _mm512_add_pd(s3, _mm512_max_pd(v, zero));
        v = _mm512_loadu_pd(p + i + 32);  s4 = _mm512_add_pd(s4, _mm512_max_pd(v, zero));
        v = _mm512_loadu_pd(p + i + 40);  s5 = _mm512_add_pd(s5, _mm512_max_pd(v, zero));
        v = _mm512_loadu_pd(p + i + 48);  s6 = _mm512_add_pd(s6, _mm512_max_pd(v, zero));
        v = _mm512_loadu_pd(p + i + 56);  s7 = _mm512_add_pd(s7, _mm512_max_pd(v, zero));
    }
    __m512d t = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(s0, s1), _mm512_add_pd(s2, s3)),
                              _mm512_add_pd(_mm512_add_pd(s4, s5), _mm512_add_pd(s6, s7)));
    __m256d lo = _mm512_extractf64x4_pd(t, 0);
    __m256d hi = _mm512_extractf64x4_pd(t, 1);
    __m256d u = _mm256_add_pd(lo, hi);               /* [a,b,c,d] */
    __m256d v = _mm256_hadd_pd(u, u);                /* [a+b, a+b, c+d, c+d] */
    __m128d w = _mm_add_pd(_mm256_castpd256_pd128(v),
                           _mm256_extractf128_pd(v, 1)); /* [total, total] */
    return _mm_cvtsd_f64(w) + tail_sum(p + i, n - i);
}

void tsvc_2_s3111_fp64(const double *restrict a, double *restrict b, int64_t LEN_1D,
                       uint8_t *restrict workspace, int64_t workspace_size)
{
    (void)workspace; (void)workspace_size;
    if (LEN_1D <= 0) { b[0] = 0.0; return; }
    if (LEN_1D < (1 << 18)) { b[0] = sum_range(a, 0, LEN_1D); return; }
    int64_t nchunks = MAX_CHUNKS;
    while (nchunks > 1 && LEN_1D / nchunks < 4096) nchunks /= 2;
    double part[MAX_CHUNKS];
#pragma omp parallel
    {
        int nt = omp_get_num_threads();
        int t = omp_get_thread_num();
        int64_t c0 = (nchunks * (int64_t)t) / nt;
        int64_t c1 = (nchunks * ((int64_t)t + 1)) / nt;
        for (int64_t c = c0; c < c1; ++c)
            part[c] = sum_range(a, c * LEN_1D / nchunks, (c + 1) * LEN_1D / nchunks);
    }
    double total = 0.0;
    for (int64_t c = 0; c < nchunks; ++c) total += part[c];
    b[0] = total;
}
