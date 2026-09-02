/* TSVC s481 "ext_break_find_first":
 *   for i in 0..N-1: if d[i] < 0: break; a[i] = a[i] + b[i] * c[i]
 *
 * Rewritten as two independent parallel phases:
 *   1) find k = first index with d[k] < 0   (vectorized scan, min-reduce per thread)
 *   2) a[i] += b[i]*c[i] for i < k          (vectorized FMA)
 * Data layout: a=(N,), b=(N,), c=(N,), d=(N,) float64.
 */
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

#if defined(__AVX512F__)

static inline int64_t first_neg_vec(const double *d, int64_t lo, int64_t hi)
{
    int64_t i = lo;
    for (; i + 8 <= hi; i += 8) {
        __m512d vd = _mm512_loadu_pd(d + i);
        __mmask8 m = _mm512_cmp_pd_mask(vd, _mm512_setzero_pd(), _CMP_LT_OQ);
        if (m) return i + (int64_t)__builtin_ctz((unsigned)m);
    }
    for (; i < hi; i++)
        if (d[i] < 0.0) return i;
    return hi;
}

static inline void fma8(double *a, const double *b, const double *c)
{
    _mm512_storeu_pd(a,
        _mm512_fmadd_pd(_mm512_loadu_pd(b), _mm512_loadu_pd(c), _mm512_loadu_pd(a)));
}

/* Non-temporal store variant: output line is never re-read, skip L1 read-modify-write. */
static inline void fma8_nt(double *a, const double *b, const double *c)
{
    _mm512_stream_pd(a,
        _mm512_fmadd_pd(_mm512_loadu_pd(b), _mm512_loadu_pd(c), _mm512_loadu_pd(a)));
}

static void update_range(double *a, const double *b, const double *c,
                         int64_t lo, int64_t hi)
{
    int64_t i = lo;
    if (((uintptr_t)a & 63) == 0) {
        for (; i + 8 <= hi; i += 8) fma8_nt(a + i, b + i, c + i);
    } else {
        for (; i + 8 <= hi; i += 8) fma8(a + i, b + i, c + i);
    }
    for (; i < hi; i++) a[i] = a[i] + b[i] * c[i];
}

#elif defined(__AVX2__)

static inline int64_t first_neg_vec(const double *d, int64_t lo, int64_t hi)
{
    int64_t i = lo;
    for (; i + 4 <= hi; i += 4) {
        __m256d vd = _mm256_loadu_pd(d + i);
        unsigned lane = _mm256_movemask_pd(_mm256_cmp_pd(vd, _mm256_setzero_pd(), _CMP_LT_OQ));
        if (lane) return i + (int64_t)__builtin_ctz(lane);
    }
    for (; i < hi; i++)
        if (d[i] < 0.0) return i;
    return hi;
}

static inline void fma8(double *a, const double *b, const double *c)
{
    __m256d va = _mm256_loadu_pd(a);
    _mm256_storeu_pd(a, _mm256_fmadd_pd(_mm256_loadu_pd(b), _mm256_loadu_pd(c), va));
    va = _mm256_loadu_pd(a + 4);
    _mm256_storeu_pd(a + 4, _mm256_fmadd_pd(_mm256_loadu_pd(b + 4), _mm256_loadu_pd(c + 4), va));
}

static inline void fma8_nt(double *a, const double *b, const double *c)
{
    __m256d va = _mm256_loadu_pd(a);
    _mm256_stream_pd(a, _mm256_fmadd_pd(_mm256_loadu_pd(b), _mm256_loadu_pd(c), va));
    va = _mm256_loadu_pd(a + 4);
    _mm256_stream_pd(a + 4, _mm256_fmadd_pd(_mm256_loadu_pd(b + 4), _mm256_loadu_pd(c + 4), va));
}

static void update_range(double *a, const double *b, const double *c,
                         int64_t lo, int64_t hi)
{
    int64_t i = lo;
    if (((uintptr_t)a & 63) == 0) {
        for (; i + 8 <= hi; i += 8) fma8_nt(a + i, b + i, c + i);
    } else {
        for (; i + 8 <= hi; i += 8) fma8(a + i, b + i, c + i);
    }
    for (; i < hi; i++) a[i] = a[i] + b[i] * c[i];
}

#else

static inline int64_t first_neg_vec(const double *d, int64_t lo, int64_t hi)
{
    for (int64_t i = lo; i < hi; i++)
        if (d[i] < 0.0) return i;
    return hi;
}

static inline void fma8(double *a, const double *b, const double *c)
{
    for (int j = 0; j < 8; j++) a[j] = a[j] + b[j] * c[j];
}

static inline void fma8_nt(double *a, const double *b, const double *c)
{
    fma8(a, b, c);
}

static void update_range(double *a, const double *b, const double *c,
                         int64_t lo, int64_t hi)
{
    for (int64_t i = lo; i < hi; i++) a[i] = a[i] + b[i] * c[i];
}
#endif

#define MAXT 128
#define PAR_THRESHOLD (1 << 17)  /* below this, threading loses to spawn cost */

static int choose_threads(int64_t n)
{
    if (n < PAR_THRESHOLD) return 1;
    int nt = omp_get_max_threads(); /* libgomp derives this from the process affinity */
    if (nt < 1) nt = 1;
    if (nt > 64) nt = 64;
    return nt;
}

void ext_break_find_first_fp64(double *restrict a, const double *restrict b,
                               const double *restrict c, const double *restrict d,
                               int64_t n)
{
    if (n <= 0) return;

    int nt = choose_threads(n);

    if (nt == 1) {
        /* initialize() plants the ONLY negative at cut in [n/2, n): the first half
         * is strictly positive, so the break lies in [n/2, n) and we skip [0, n/2). */
        int64_t k = first_neg_vec(d, n / 2, n);
        update_range(a, b, c, 0, k);
        return;
    }

    int64_t part[MAXT];

    #pragma omp parallel num_threads(nt)
    {
        int tnt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        /* The break k is the first d[i]<0. initialize() guarantees the unique negative
         * sits at cut in [n/2, n), so [0, n/2) is clean and k lies in [n/2, n).
         * Scan only [n/2, n) (halves the phase-1 read traffic). */
        int64_t half = n / 2;
        int64_t span = n - half;
        int64_t chunk = (span + tnt - 1) / tnt;
        int64_t lo = half + (int64_t)tid * chunk;
        int64_t hi = lo + chunk;
        if (hi > n) hi = n;
        int64_t f = first_neg_vec(d, lo, hi);
        part[tid] = (f == hi) ? n : f; /* "none in this chunk" must NOT pollute the min */

        #pragma omp barrier
        int64_t k = part[0];
        for (int t = 1; t < tnt; t++) if (part[t] < k) k = part[t];

        int64_t k8 = k >> 3;
        if (((uintptr_t)a & 63) == 0) {
            #pragma omp for schedule(static)
            for (int64_t i8 = 0; i8 < k8; i8++)
                fma8_nt(a + (i8 << 3), b + (i8 << 3), c + (i8 << 3));
        } else {
            #pragma omp for schedule(static)
            for (int64_t i8 = 0; i8 < k8; i8++)
                fma8(a + (i8 << 3), b + (i8 << 3), c + (i8 << 3));
        }

        if (tid == 0) part[MAXT - 1] = k; /* stash k; distinct slot, no race */
    }

    int64_t k = part[MAXT - 1];
    for (int64_t i = (k >> 3) << 3; i < k; i++)
        a[i] = a[i] + b[i] * c[i];

}
