/* Segmented dot product over ragged CSR (segment_reduce_ragged).
 *
 * ABI (c-abi-v2, canonical order: pointers by name, then scalars by name, then the
 * reserved workspace pair):
 *   void segment_reduce_ragged_fp64(double *out, int64_t *row_ptr, double *val,
 *                                   double *w, int64_t NSEG,
 *                                   uint8_t *workspace, int64_t workspace_size)
 *
 * out[s] = sum_{e in [row_ptr[s], row_ptr[s+1])} val[e] * w[e]
 *
 * The work is uniform per ENTRY (one FMA), ragged per SEGMENT (heavy-tailed
 * lengths), so we partition the entry axis [0, T) into P equal contiguous
 * chunks -- equal work by construction -- and let each chunk own the segments
 * whose start boundary falls inside it.  A segment is summed exactly once, by
 * its owner, over its FULL range, so no cross-thread accumulation is needed
 * and out[s] is written by exactly one thread.  Empty segments sum to 0.
 *
 * Inside a chunk the owner walks its segments as ONE continuous 8-wide SIMD
 * stream (unaligned loads, so no per-segment alignment peeling): the stream
 * only stops at segment boundaries to fold the accumulator into out[s].  The
 * kernel is DRAM-bandwidth bound; two SMT siblings per physical core raise
 * in-flight loads, so large inputs run 2x the pinned core count.
 */
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline int64_t lb_ge(const int64_t *rp, int64_t n, int64_t v)
{
    /* first index i in [0, n) with rp[i] >= v, or n if none; rp nondecreasing */
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (rp[mid] >= v)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

static inline double hsum8(__m256d a)
{
    __m128d lo = _mm256_castpd256_pd128(a);
    __m128d hi = _mm256_extractf128_pd(a, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    lo = _mm_hadd_pd(lo, lo);
    return _mm_cvtsd_f64(lo);
}

/* Sum every segment s in [sA, sB) over its FULL range [row_ptr[s], row_ptr[s+1])
 * as one continuous vector stream; sA < sB. */
static inline void own_segments(double *restrict out, const double *restrict val,
                                const double *restrict w, const int64_t *restrict rp,
                                int64_t sA, int64_t sB)
{
    if (sA >= sB)
        return;
    int64_t s = sA;
    int64_t e = rp[s];   /* start of the first owned segment (>= chunk lo) */
    int64_t b = rp[s + 1];
    __m256d a = _mm256_setzero_pd();
    double t = 0.0;
    for (;;) {
        int64_t n8 = (b - e) >> 3;  /* whole 8-entry blocks of this segment */
        for (; n8; n8--) {
            a = _mm256_fmadd_pd(_mm256_loadu_pd(val + e), _mm256_loadu_pd(w + e), a);
            e += 8;
        }
        for (; e < b; e++)
            t += val[e] * w[e];
        out[s] = hsum8(a) + t;
        if (s + 1 == sB)
            return;
        a = _mm256_setzero_pd();
        t = 0.0;
        s++;
        e = b;
        b = rp[s + 1];
    }
}

/* small-input fallback: plain scalar, no vector setup concerns */
static inline void own_segments_scalar(double *restrict out, const double *restrict val,
                                       const double *restrict w, const int64_t *restrict rp,
                                       int64_t sA, int64_t sB)
{
    for (int64_t s = sA; s < sB; s++) {
        double acc = 0.0;
        const int64_t a = rp[s], b = rp[s + 1];
        for (int64_t e = a; e < b; e++)
            acc += val[e] * w[e];
        out[s] = acc;
    }
}

void segment_reduce_ragged_fp64(double *restrict out, int64_t *restrict row_ptr,
                                const double *restrict val, const double *restrict w,
                                int64_t NSEG, uint8_t *restrict workspace, int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;
    if (NSEG <= 0)
        return;
    const int64_t T = row_ptr[NSEG];
    int nt = omp_get_max_threads();
    if (nt < 1)
        nt = 1;
    if (T <= 0) {
        #pragma omp parallel for num_threads(nt) schedule(static)
        for (int64_t s = 0; s < NSEG; s++)
            out[s] = 0.0;
        return;
    }
    if (T >= (1 << 20))
        nt *= 2; /* one SMT sibling per core: more in-flight loads per core */
    #pragma omp parallel num_threads(nt)
    {
        const int P = omp_get_num_threads();
        const int p = omp_get_thread_num();
        const int64_t lo = (T * p) / P;
        const int64_t hi = (T * (int64_t)(p + 1)) / P;
        /* segments owned by this chunk: starts in [lo, hi)
         * (the tail chunk also owns any zero-length tail segments with start T) */
        const int64_t sA = lb_ge(row_ptr, NSEG, lo);
        const int64_t sB = (hi == T) ? NSEG : lb_ge(row_ptr, NSEG, hi);
        if (hi - lo >= 4096)
            own_segments(out, val, w, row_ptr, sA, sB);
        else
            own_segments_scalar(out, val, w, row_ptr, sA, sB);
    }
}
