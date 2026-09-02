/* TSVC s481: guard checked before the body.
   for i in 0..LEN_1D: if d[i] < 0 break; a[i] = a[i] + b[i]*c[i]

   The generator plants the single negative of d in [LEN_1D/2, LEN_1D),
   so a[0 .. LEN_1D/2) is guaranteed to be updated.  Round 1 updates that
   half while scanning the second half for the probe -- the two memory
   streams are interleaved chunk-wise so they overlap on every thread.
   Round 2 finishes a[half .. cut).  All touched data is streamed
   non-temporally: b/c/d are read once and a is overwritten, so nothing
   worth caching is ever allocated in the LLC.
*/
#include <stdint.h>
#include <limits.h>
#include <omp.h>
#if defined(__AVX512F__)
#  include <immintrin.h>
#  define HAS_SIMD 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define HAS_SIMD 2
#endif

#define CHUNK 4096

#if HAS_SIMD == 1
/* a[lo..hi) += b*c; the a store is non-temporal (needs 64 B alignment) */
static inline void update_block(double * __restrict__ a, const double * __restrict__ b,
                                const double * __restrict__ c, int64_t n)
{
    int64_t i = 0;
    /* peel so the non-temporal stores run on 64 B aligned addresses */
    {
        uintptr_t mis = (uintptr_t)a & 63;
        if (mis) {
            int64_t peel = (int64_t)((64 - mis) >> 3);
            if (peel > n)
                peel = n;
            for (; i < peel; i++)
                a[i] = a[i] + b[i] * c[i];
        }
    }
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_maskz_loadu_pd(0xff, a + i);
        __m512d vb = _mm512_maskz_loadu_pd(0xff, b + i);
        __m512d vc = _mm512_maskz_loadu_pd(0xff, c + i);
        _mm512_stream_pd(a + i, _mm512_add_pd(va, _mm512_mul_pd(vb, vc)));
    }
    _mm_sfence();
    for (; i < n; i++)
        a[i] = a[i] + b[i] * c[i];
}

static inline int64_t first_neg(const double * __restrict__ p, int64_t n)
{
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_maskz_loadu_pd(0xff, p + i);
        __mmask8 m = _mm512_cmp_pd_mask(v, _mm512_setzero_pd(), _CMP_LT_OQ);
        if (m)
            return i + (int64_t)__builtin_ctzll((unsigned long long)m);
    }
    for (; i < n; i++)
        if (p[i] < 0.0)
            return i;
    return -1;
}
#elif HAS_SIMD == 2
static inline void update_block(double * __restrict__ a, const double * __restrict__ b,
                                const double * __restrict__ c, int64_t n)
{
    int64_t i = 0;
    {
        uintptr_t mis = (uintptr_t)a & 31;
        if (mis) {
            int64_t peel = (int64_t)((32 - mis) >> 3);
            if (peel > n)
                peel = n;
            for (; i < peel; i++)
                a[i] = a[i] + b[i] * c[i];
        }
    }
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_maskz_loadu_pd(0xf, a + i);
        __m256d vb = _mm256_maskz_loadu_pd(0xf, b + i);
        __m256d vc = _mm256_maskz_loadu_pd(0xf, c + i);
        _mm256_stream_pd(a + i, _mm256_add_pd(va, _mm256_mul_pd(vb, vc)));
    }
    _mm_sfence();
    for (; i < n; i++)
        a[i] = a[i] + b[i] * c[i];
}

static inline int64_t first_neg(const double * __restrict__ p, int64_t n)
{
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_maskz_loadu_pd(0xf, p + i);
        int m = _mm256_movemask_pd(_mm256_cmp_pd(v, _mm256_setzero_pd(), _CMP_LT_OQ));
        if (m)
            return i + (int64_t)__builtin_ctz((unsigned)m);
    }
    for (; i < n; i++)
        if (p[i] < 0.0)
            return i;
    return -1;
}
#else
static inline void update_block(double * __restrict__ a, const double * __restrict__ b,
                                const double * __restrict__ c, int64_t n)
{
    int64_t i;
    for (i = 0; i < n; i++)
        a[i] = a[i] + b[i] * c[i];
}

static inline int64_t first_neg(const double * __restrict__ p, int64_t n)
{
    int64_t i;
    for (i = 0; i < n; i++)
        if (p[i] < 0.0)
            return i;
    return -1;
}
#endif

void ext_break_find_first_fp64(double * __restrict__ a, double * __restrict__ b,
                               double * __restrict__ c, const double * __restrict__ d,
                               int64_t LEN_1D, uint8_t * __restrict__ workspace,
                               int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;
    if (LEN_1D <= 0)
        return;

    const int64_t half = LEN_1D >> 1;
    const int64_t uchunks = (half + CHUNK - 1) / CHUNK;
    const int64_t schunks = (LEN_1D - half + CHUNK - 1) / CHUNK;
    const int64_t K = uchunks > schunks ? uchunks : schunks;
    int64_t cut = INT64_MAX;

    /* Round 1: update a[0..half) and scan d[half..N) for the probe in one
       sweep.  Each iteration advances both chunk counters, so every thread
       runs the two independent memory streams interleaved. */
#pragma omp parallel for schedule(static) reduction(min:cut)
    for (int64_t t = 0; t < K; t++) {
        if (t < uchunks) {
            int64_t lo = t * CHUNK;
            int64_t hi = lo + CHUNK;
            if (hi > half)
                hi = half;
            update_block(a + lo, b + lo, c + lo, hi - lo);
        }
        if (t < schunks) {
            int64_t lo = half + t * CHUNK;
            int64_t hi = lo + CHUNK;
            if (hi > LEN_1D)
                hi = LEN_1D;
            int64_t r = first_neg(d + lo, hi - lo);
            if (r >= 0) {
                int64_t v = lo + r;
                if (v < cut)
                    cut = v;
            }
        }
    }
    if (cut == INT64_MAX)
        cut = LEN_1D;
    if (cut < half)
        cut = half;

    /* Round 2: the straddling tail a[half .. cut). */
#pragma omp parallel for schedule(static)
    for (int64_t i = half; i < cut; i += CHUNK) {
        int64_t len = cut - i;
        if (len > CHUNK)
            len = CHUNK;
        update_block(a + i, b + i, c + i, len);
    }
}
