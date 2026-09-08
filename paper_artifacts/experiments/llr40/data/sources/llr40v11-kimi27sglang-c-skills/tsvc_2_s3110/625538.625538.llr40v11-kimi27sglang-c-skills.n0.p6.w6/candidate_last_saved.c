#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

typedef struct {
    double v;
    int64_t k;
} maxloc_t;

static inline maxloc_t combine_maxloc(maxloc_t a, maxloc_t b) {
    if (a.v > b.v) return a;
    if (b.v > a.v) return b;
    return (a.k <= b.k) ? a : b;
}

#pragma omp declare reduction(maxloc : maxloc_t : omp_out = combine_maxloc(omp_out, omp_in)) \
    initializer(omp_priv = {-INFINITY, -1})

static inline maxloc_t reduce_m256(__m256d vmax, __m256i kmax) {
    double vals[4];
    int64_t ks[4];
    _mm256_storeu_pd(vals, vmax);
    _mm256_storeu_si256((__m256i *)ks, kmax);

    maxloc_t best = {vals[0], ks[0]};
    for (int t = 1; t < 4; ++t)
        best = combine_maxloc(best, (maxloc_t){vals[t], ks[t]});
    return best;
}

void tsvc_2_s3110_fp64(const double *restrict aa, double *restrict bb, const int64_t LEN_2D) {
    const int64_t n = LEN_2D;
    const int64_t total = n * n;
    maxloc_t m = {-INFINITY, -1};

    #pragma omp parallel reduction(maxloc : m)
    {
        __m256d vmax = _mm256_set1_pd(-INFINITY);
        __m256i kmax = _mm256_set1_epi64x(-1);
        maxloc_t tbest = {-INFINITY, -1};

        #pragma omp for schedule(static) nowait
        for (int64_t k = 0; k <= total - 4; k += 4) {
            __m256d v = _mm256_loadu_pd(aa + k);
            __m256d gt = _mm256_cmp_pd(v, vmax, _CMP_GT_OQ);
            vmax = _mm256_blendv_pd(vmax, v, gt);
            __m256i idxv = _mm256_set_epi64x(k + 3, k + 2, k + 1, k + 0);
            kmax = _mm256_blendv_epi8(kmax, idxv, _mm256_castpd_si256(gt));
        }

        #pragma omp for schedule(static) nowait
        for (int64_t k = total & ~((int64_t)3); k < total; ++k) {
            double v = aa[k];
            if (v > tbest.v) {
                tbest.v = v;
                tbest.k = k;
            }
        }

        maxloc_t vec = reduce_m256(vmax, kmax);
        m = combine_maxloc(vec, tbest);
    }

    int64_t i = m.k / n;
    int64_t j = m.k % n;
    bb[0] = m.v + (double)i + (double)j;
}
