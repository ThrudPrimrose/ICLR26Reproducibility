#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

typedef struct { double v; int64_t i; } mx;

/* left-to-right fold: replace iff strictly greater (serial-scan semantics;
 * NaN never beats anything and is only "kept" as the left operand) */
static inline mx merge_lr(mx a, mx b)
{
    if (b.v > a.v) return b;
    if (a.v > b.v) return a;
    int64_t i = a.i < b.i ? a.i : b.i;
    return (mx){a.v, i};
}

static inline double hmax256(__m256d v)
{
    __m128d h = _mm_max_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
    return _mm_cvtsd_f64(_mm_max_pd(_mm_unpacklo_pd(h, h), _mm_unpackhi_pd(h, h)));
}

static inline int eqpos(__m256d v, double cm)
{
    return __builtin_ctz(_mm256_movemask_pd(
        _mm256_cmp_pd(v, _mm256_set1_pd(cm), _CMP_EQ_OQ)));
}

static inline void consider(mx *st, __m256d v, int64_t off)
{
    int m = _mm256_movemask_pd(_mm256_cmp_pd(v, _mm256_set1_pd(st->v), _CMP_GT_OQ));
    if (m) {
        double cm = hmax256(v);
        st->v = cm;
        st->i = off + eqpos(v, cm);
    }
}

/* scan a[off .. off+n): returns (max value, first index where it occurs) */
static mx scan_region(const double *a, int64_t off, int64_t n)
{
    const double *p = a + off;
    mx st = {p[0], off};
    int64_t i = 0, n4 = n & ~(int64_t)3;
    for (; i + 16 <= n4; i += 16) {
        __m256d v0 = _mm256_loadu_pd(p + i);
        __m256d v1 = _mm256_loadu_pd(p + i + 4);
        __m256d v2 = _mm256_loadu_pd(p + i + 8);
        __m256d v3 = _mm256_loadu_pd(p + i + 12);
        consider(&st, v0, off + i);
        consider(&st, v1, off + i + 4);
        consider(&st, v2, off + i + 8);
        consider(&st, v3, off + i + 12);
    }
    for (; i < n4; i += 4) {
        __m256d v = _mm256_loadu_pd(p + i);
        consider(&st, v, off + i);
    }
    for (; i < n; i++)
        if (p[i] > st.v) { st.v = p[i]; st.i = off + i; }
    return st;
}

#define PAR_THRESHOLD (1 << 19)

void argmax_with_index_fp64(double *a, int64_t *out_index, double *out_value,
                            int64_t LEN_1D, uint8_t *ws, int64_t ws_bytes)
{
    (void)ws; (void)ws_bytes;
    int64_t n = LEN_1D;
    if (n <= 1) { *out_value = a[0]; *out_index = 0; return; }

    mx best;
    if (n < PAR_THRESHOLD) {
        best = scan_region(a, 0, n);
    } else {
        int nt = omp_get_max_threads();
        if (nt > (int)n) nt = (int)n;
        mx parts[256];
        #pragma omp parallel num_threads(nt)
        {
            int t = omp_get_thread_num();
            int64_t lo = (n * t) / nt;
            int64_t hi = (n * (t + 1)) / nt;
            parts[t] = scan_region(a, lo, hi - lo);
        }
        best = parts[0];
        for (int t = 1; t < nt; t++) best = merge_lr(best, parts[t]);
    }
    *out_value = best.v;
    *out_index = best.i;
}
