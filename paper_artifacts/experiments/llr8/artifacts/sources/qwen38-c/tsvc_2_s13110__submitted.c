/* TSVC_2 s13110 -- max of aa (LEN_2D x LEN_2D, row-major) plus the FIRST
 * (row-major) location of that max;  bb[0,0] = maxv + xindex + yindex.
 *
 * The numpy oracle is a left-to-right scan with strict `>` updates, which is
 * exactly the commutative/associative reduction "max value, min flat index on
 * ties" (a NaN element never wins, mirroring `x > maxv` being false for NaN;
 * aa[0,0] == NaN is special-cased below, as in the oracle).  That makes one
 * streaming pass over aa fully parallel: each core reduces its chunk with an
 * AVX-512 4-chain lexicographic (value, index) fold, then the partials are
 * combined serially in O(nt).
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <omp.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#define TMAX 256
static double  g_pmax[TMAX];
static int64_t g_pidx[TMAX];

/* fold candidate (v, idx) into (m, k): larger value wins; on an exact tie the
 * smaller index wins.  Callers guarantee v is not NaN. */
static inline void lex_fold(double *m, int64_t *k, double v, int64_t idx)
{
    if (v > *m) { *m = v; *k = idx; }
    else if (v == *m && idx < *k) *k = idx;
}

#if defined(__AVX512F__)
/* Reduce a[lo,hi) to (max value, first index of that max), left to right.
 *
 * Keep the running max as an 8-lane broadcast `maxvec`.  For every 8-block, a
 * single vector compare finds whether ANY lane is a new record; only in that
 * rare case (O(log n) records per chunk) do we horizontal-max the block and
 * pick the first lane equal to it.  Between records maxvec is constant, so the
 * per-block compares are independent and stream at memory speed.  NaNs never
 * compare greater, matching the oracle's strict `>`.
 *
 * The body is a double-buffered 8x8 stream: while the current 8 blocks are
 * resolved, the next 8 loads are already in flight -> ~16 outstanding lines
 * per core, which saturates DRAM bandwidth. */
#define REC(c, idx)                                                             \
    do {                                                                        \
        if (_mm512_cmp_pd_mask((c), maxvec, _CMP_GT_OQ)) {                      \
            maxv = _mm512_reduce_max_pd(c);                                     \
            maxvec = _mm512_set1_pd(maxv);                                      \
            bestk = (int64_t)(idx) +                                            \
                   _tzcnt_u32(_mm512_cmp_pd_mask(c, maxvec, _CMP_EQ_OQ));       \
        }                                                                       \
    } while (0)

static void scan(const double *restrict a, int64_t lo, int64_t hi,
                 double *mout, int64_t *kout)
{
    double   maxv  = -INFINITY;
    int64_t  bestk = lo;
    __m512d  maxvec = _mm512_set1_pd(-INFINITY);
    int64_t  q = lo;

    const int64_t vhi64 = lo + ((hi - lo) / 64) * 64;
    if (vhi64 > lo) {
        __m512d c0 = _mm512_loadu_pd(a + lo + 0),  c1 = _mm512_loadu_pd(a + lo + 8);
        __m512d c2 = _mm512_loadu_pd(a + lo + 16), c3 = _mm512_loadu_pd(a + lo + 24);
        __m512d c4 = _mm512_loadu_pd(a + lo + 32), c5 = _mm512_loadu_pd(a + lo + 40);
        __m512d c6 = _mm512_loadu_pd(a + lo + 48), c7 = _mm512_loadu_pd(a + lo + 56);
        for (; q + 128 <= vhi64; q += 64) {
            __m512d d0 = _mm512_loadu_pd(a + q + 64),  d1 = _mm512_loadu_pd(a + q + 72);
            __m512d d2 = _mm512_loadu_pd(a + q + 80),  d3 = _mm512_loadu_pd(a + q + 88);
            __m512d d4 = _mm512_loadu_pd(a + q + 96),  d5 = _mm512_loadu_pd(a + q + 104);
            __m512d d6 = _mm512_loadu_pd(a + q + 112), d7 = _mm512_loadu_pd(a + q + 120);
            REC(c0, q + 0);  REC(c1, q + 8);  REC(c2, q + 16); REC(c3, q + 24);
            REC(c4, q + 32); REC(c5, q + 40); REC(c6, q + 48); REC(c7, q + 56);
            c0 = d0; c1 = d1; c2 = d2; c3 = d3; c4 = d4; c5 = d5; c6 = d6; c7 = d7;
        }
        REC(c0, q + 0);  REC(c1, q + 8);  REC(c2, q + 16); REC(c3, q + 24);
        REC(c4, q + 32); REC(c5, q + 40); REC(c6, q + 48); REC(c7, q + 56);
        q += 64;
    }

    for (; q + 8 <= hi; q += 8) {
        __m512d c = _mm512_loadu_pd(a + q);
        REC(c, q);
    }
    for (; q < hi; q++) {
        double v = a[q];
        if (v > maxv) { maxv = v; bestk = q; }
    }
    *mout = maxv;
    *kout = bestk;
}
#undef REC
#else
static void scan(const double *restrict a, int64_t lo, int64_t hi,
                 double *mout, int64_t *kout)
{
    double m = -INFINITY;
    int64_t k = INT64_MAX;
    for (int64_t q = lo; q < hi; q++) {
        double v = a[q];
        if (v == v) lex_fold(&m, &k, v, q);
    }
    *mout = m;
    *kout = k;
}
#endif

void tsvc_2_s13110_fp64(double *restrict aa, double *restrict bb,
                        int64_t LEN_2D, void *workspace, int64_t workspace_bytes)
{
    (void)workspace; (void)workspace_bytes;
    const int64_t N = LEN_2D * LEN_2D;

    /* oracle: maxv starts at aa[0,0]; if that is NaN, `x > NaN` is never true,
     * so the answer is exactly (NaN, 0, 0). */
    if (N == 0) { bb[0] = 0.0; return; }
    if (aa[0] != aa[0]) { bb[0] = aa[0]; return; }

    int nt = (int)omp_get_max_threads();
    if (nt < 1) nt = 1;
    if (nt > TMAX) nt = TMAX;

    double maxv = -INFINITY;
    int64_t kbest = INT64_MAX;

    if (N >= (1 << 20) && nt > 1) {
        int64_t chunk = (N + nt - 1) / nt;
#pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            int64_t lo = (int64_t)tid * chunk;
            int64_t hi = lo + chunk;
            if (hi > N) hi = N;
            double m; int64_t k;
            scan(aa, lo, hi, &m, &k);
            g_pmax[tid] = m;
            g_pidx[tid] = k;
        }
        for (int t = 0; t < nt; t++) lex_fold(&maxv, &kbest, g_pmax[t], g_pidx[t]);
    } else {
        scan(aa, 0, N, &maxv, &kbest);
    }

    int64_t xi = kbest / LEN_2D;
    int64_t yi = kbest % LEN_2D;
    bb[0] = (maxv + (double)xi) + (double)yi;
}
