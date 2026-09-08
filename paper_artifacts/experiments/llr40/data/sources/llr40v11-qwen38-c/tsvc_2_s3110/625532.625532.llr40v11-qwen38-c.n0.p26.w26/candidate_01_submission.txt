#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

/* tsvc_2 s3110: first row-major occurrence of the fp64 argmax over an N x N
 * matrix; bb[0] = maxv + xindex + yindex.
 * Memory-bound single-pass reduction: OpenMP contiguous row-major chunks,
 * AVX2 4-wide scan with first-occurrence lane tracking. */

#define MAXT 1024

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t n2 = LEN_2D * LEN_2D;
    static double thr_max[MAXT];
    static int64_t thr_pos[MAXT];

    #pragma omp parallel
    {
        int64_t tid = omp_get_thread_num();
        int64_t nt = omp_get_num_threads();

        /* contiguous partition */
        int64_t base = n2 / nt;
        int64_t rem = n2 % nt;
        int64_t start, end;
        if (tid < rem) { start = tid * (base + 1); end = start + base + 1; }
        else           { start = rem * (base + 1) + (tid - rem) * base; end = start + base; }

        double best = -INFINITY;
        int64_t best_pos = -1;
        const double *p = aa + start;
        const double *lim = aa + end;

        while (p + 4 <= lim) {
            __m256d v = _mm256_loadu_pd(p);
            __m128d hi = _mm256_extractf128_pd(v, 1);
            __m128d lo = _mm256_castpd256_pd128(v);
            __m128d m1 = _mm_max_pd(lo, hi);
            double mx = _mm_cvtsd_f64(_mm_max_pd(m1, _mm_unpackhi_pd(m1, m1)));
            if (mx > best) {
                int mask = _mm256_movemask_pd(_mm256_cmp_pd(v, _mm256_set1_pd(mx), _CMP_EQ_OQ));
                best = mx;
                best_pos = (int64_t)(p - aa) + __builtin_ctz(mask);
            }
            p += 4;
        }
        while (p < lim) {
            double v = *p;
            if (v > best) { best = v; best_pos = (int64_t)(p - aa); }
            p++;
        }
        if (tid < MAXT) {
            thr_max[tid] = best;
            thr_pos[tid] = best_pos;
        }
    }

    int nt = omp_get_max_threads();
    double gmax = -INFINITY;
    int64_t gpos = -1;
    for (int t = 0; t < nt && t < MAXT; ++t) {
        int64_t pos = thr_pos[t];
        if (pos < 0) continue;
        double m = thr_max[t];
        if (m > gmax || (m == gmax && pos < gpos)) { gmax = m; gpos = pos; }
    }
    bb[0] = gmax + (double)(gpos / LEN_2D) + (double)(gpos % LEN_2D);
}
