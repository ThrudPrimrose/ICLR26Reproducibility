/* wf_triangular optimized: banded anti-diagonal wavefront + SIMD prefix scans.
 *
 * a[i][j] (j>=i) depends on a[i-1][j] and a[i][j-1] -- the previous anti-diagonal
 * (s = i+j).  A band of B anti-diagonals [s, s+B) is one parallel region; within the
 * band, row i covers the consecutive column run [js, je] (race-free: row i only reads
 * row i-1, finished earlier in the band, and its own previous column), so each row is a
 * prefix scan  c_j = b_j + c_{j-1},  b_j = a[i][j] + a[i-1][j]
 * evaluated with a wide-shuffle masked SIMD butterfly prefix (8 lanes AVX512, 4 AVX2).
 *
 * NOTE: per-element value follows ((a[i][j]+a[i-1][j]) + a[i][j-1]); the in-block
 * butterfly sums b with tree (not left-to-right) grouping, i.e. within ~1 ulp of the
 * sequential reference. */

#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>
#include <omp.h>

#define WF_B 128

#define WSHUF(T, IMM) __extension__ ({ typeof(T) __r; \
    __asm__("vpermpd $" #IMM ", %1, %0" : "=v"(__r) : "v"(T)); __r; })

#ifdef __AVX512F__
static inline void row_scan(double *restrict row, const double *restrict prev,
                            int64_t j0, int64_t j1)
{
    double carry = row[j0 - 1];
    int64_t j = j0;
    const __mmask8 m1 = 0xFE, m2 = 0xFC, m3 = 0xF0;
    __m512d vc = _mm512_set1_pd(carry);
    for (; j + 8 <= j1 + 1; j += 8) {
        __m512d b = _mm512_add_pd(_mm512_loadu_pd(row + j), _mm512_loadu_pd(prev + j));
        __m512d t = _mm512_mask_add_pd(b, m1, b, WSHUF(b, 0x24));
        t = _mm512_mask_add_pd(t, m2, t, WSHUF(t, 0x14));
        t = _mm512_mask_add_pd(t, m3, t, WSHUF(t, 0x64));
        t = _mm512_add_pd(t, vc);
        _mm512_storeu_pd(row + j, t);
        carry = _mm_cvtsd_f64(_mm_shuffle_pd(_mm512_extractf64x2_pd(t, 3),
                                             _mm512_extractf64x2_pd(t, 3), 1));
        vc = _mm512_set1_pd(carry);
    }
    for (; j <= j1; j++) {
        row[j] = row[j] + prev[j] + carry;
        carry = row[j];
    }
}
#elif defined(__AVX2__)
static inline void row_scan(double *restrict row, const double *restrict prev,
                            int64_t j0, int64_t j1)
{
    double carry = row[j0 - 1];
    int64_t j = j0;
    const __m256d kM1 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_set_pd(0., 1.)),
                                             _mm_set1_pd(1.), 0);
    const __m256d kM2 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_setzero_pd()),
                                             _mm_set1_pd(1.), 0);
    __m256d vc = _mm256_set1_pd(carry);
    for (; j + 4 <= j1 + 1; j += 4) {
        __m256d b = _mm256_add_pd(_mm256_loadu_pd(row + j), _mm256_loadu_pd(prev + j));
        __m256d t = _mm256_add_pd(b, _mm256_and_pd(WSHUF(b, 0x24), kM1));
        t = _mm256_add_pd(t, _mm256_and_pd(WSHUF(t, 0x14), kM2));
        t = _mm256_add_pd(t, vc);
        _mm256_storeu_pd(row + j, t);
        carry = _mm_cvtsd_f64(_mm256_extractf128_pd(t, 1));
        vc = _mm256_set1_pd(carry);
    }
    for (; j <= j1; j++) {
        row[j] = row[j] + prev[j] + carry;
        carry = row[j];
    }
}
#else
static inline void row_scan(double *restrict row, const double *restrict prev,
                            int64_t j0, int64_t j1)
{
    double carry = row[j0 - 1];
    for (int64_t j = j0; j <= j1; j++) {
        row[j] = row[j] + prev[j] + carry;
        carry = row[j];
    }
}
#endif

void wf_triangular_fp64(double *restrict a, const int64_t N)
{
    if (N < 2) return;
    const int64_t s_max = 2 * N - 2;
#pragma omp parallel
    {
        for (int64_t s = 2; s <= s_max; s += WF_B) {
            int64_t s1 = s + WF_B - 1;
            if (s1 > s_max) s1 = s_max;
            int64_t i_lo = s - N + 1;
            if (i_lo < 1) i_lo = 1;
            int64_t i_hi = s1 / 2;
            if (i_hi > N - 1) i_hi = N - 1;
#pragma omp for schedule(static)
            for (int64_t i = i_lo; i <= i_hi; i++) {
                int64_t js = s > 2 * i ? s : 2 * i;
                int64_t je = s1 < i + N - 1 ? s1 : i + N - 1;
                row_scan(a + i * N, a + (i - 1) * N, js - i, je - i);
            }
        }
    }
}
