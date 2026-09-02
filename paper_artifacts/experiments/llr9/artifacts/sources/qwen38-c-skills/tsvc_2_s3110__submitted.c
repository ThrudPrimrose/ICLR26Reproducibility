#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

typedef struct {
    double maxv;
    int64_t idx; /* flat (row-major) index of first occurrence of maxv in span */
} part_t;

/* One-pass max-with-first-index over p[0..n). Out of pure streaming loads:
   per-lane running max, and per-lane index of that max (exact doubles, n < 2^53). */
static inline void scan_span(const double *restrict p, int64_t n, double *outmax, int64_t *outidx)
{
    const __m256d vlane = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
    const __m256d vfour = _mm256_set1_pd(4.0);
    __m256d vmax = _mm256_set1_pd(-INFINITY);
    __m256d vidx = _mm256_setzero_pd();
    __m256d vbase = _mm256_setzero_pd();
    int64_t k = 0;
    const int64_t nv = n & ~(int64_t)3;
    for (; k < nv; k += 4) {
        __m256d v = _mm256_loadu_pd(p + k);
        __m256d mv = _mm256_cmp_pd(v, vmax, _CMP_GT_OQ); /* sign-bit mask */
        vmax = _mm256_max_pd(vmax, v);
        vidx = _mm256_blendv_pd(vidx, _mm256_add_pd(vbase, vlane), mv);
        vbase = _mm256_add_pd(vbase, vfour);
    }
    __m256d vfull = vmax; /* per-lane maxima before horizontal reduction */
    __m256d s = _mm256_shuffle_pd(vmax, vmax, 5);
    vmax = _mm256_max_pd(vmax, s);
    s = _mm256_permute2f128_pd(vmax, vmax, 0x11);
    vmax = _mm256_max_pd(vmax, s);
    const double pm = _mm_cvtsd_f64(_mm256_castpd256_pd128(vmax));
    __m256d eq = _mm256_cmp_pd(vfull, _mm256_set1_pd(pm), _CMP_EQ_OQ);
    __m256d cand = _mm256_blendv_pd(_mm256_set1_pd(1e300), vidx, eq);
    s = _mm256_shuffle_pd(cand, cand, 5);
    cand = _mm256_min_pd(cand, s);
    s = _mm256_permute2f128_pd(cand, cand, 0x11);
    cand = _mm256_min_pd(cand, s);
    const double pi = _mm_cvtsd_f64(_mm256_castpd256_pd128(cand));
    *outmax = pm;
    *outidx = (int64_t)pi;
    for (; k < n; ++k) {
        if (p[k] > *outmax) {
            *outmax = p[k];
            *outidx = k;
        }
    }
}

void tsvc_2_s3110_fp64(
    const double *restrict aa,
    double *restrict bb,
    const int64_t LEN_2D,
    uint8_t *restrict workspace,
    const int64_t workspace_size)
{
    const int64_t n = LEN_2D * LEN_2D;
    double gmax;
    int64_t gidx;

    if (n <= 4096) {
        /* Small case (e.g. S preset, 32 KiB): data fits L1/L2, a thread
           team's fork-join would cost more than it saves. Serial scan. */
        scan_span(aa, n, &gmax, &gidx);
    } else {
        const int nt = omp_get_max_threads();
        const size_t need = (size_t)nt * sizeof(part_t);
        size_t asz = (need + 63u) & ~(size_t)63u;
        part_t *parts;
        int usews = (workspace != NULL) && (workspace_size >= (int64_t)need);
        if (usews) {
            parts = (part_t *)workspace;
        } else {
            parts = (part_t *)aligned_alloc(64, asz);
        }
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int64_t nt64 = (int64_t)omp_get_num_threads();
            int64_t chunk = (n + nt64 - 1) / nt64;
            int64_t lo = (int64_t)tid * chunk;
            int64_t hi = lo + chunk;
            if (hi > n)
                hi = n;
            if (lo < hi) {
                double m;
                int64_t ix;
                scan_span(aa + lo, hi - lo, &m, &ix);
                parts[tid].maxv = m;
                parts[tid].idx = lo + ix;
            } else {
                parts[tid].maxv = -INFINITY;
                parts[tid].idx = lo;
            }
        }
        gmax = parts[0].maxv;
        gidx = parts[0].idx;
        for (int t = 1; t < nt; ++t) {
            if (parts[t].maxv > gmax) {
                gmax = parts[t].maxv;
                gidx = parts[t].idx;
            } else if (parts[t].maxv == gmax && parts[t].idx < gidx) {
                gidx = parts[t].idx;
            }
        }
        if (!usews)
            free(parts);
    }

    const int64_t i = gidx / LEN_2D;
    const int64_t j = gidx - i * LEN_2D;
    bb[0] = gmax + (double)i + (double)j;
}
