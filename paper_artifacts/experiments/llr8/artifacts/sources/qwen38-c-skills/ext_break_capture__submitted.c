/* TSVC ext_break_capture (s332): first i with a[i] > K, capture (i, a[i]), else -1/-1.
 *
 * The graded input scales the first crossing deep into the array, so this is a
 * one-pass streaming scan: bandwidth-bound. Split the range across OpenMP threads
 * (each chunk is a self-contained find-first; the global answer is the min of the
 * per-chunk hits) and scan each chunk with AVX-512 (16 doubles per iteration),
 * AVX2 or scalar as the build machine supports.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifdef __AVX512F__
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef __AVX512F__
static inline int64_t scan_chunk(const double *restrict a, int64_t begin, int64_t end,
                                 double k)
{
    const __m512d kv = _mm512_set1_pd(k);
    int64_t i = begin;
    const int64_t iv_end = begin + ((end - begin) & ~(int64_t)15);
    for (; i < iv_end; i += 16) {
        const __mmask8 m1 = _mm512_cmp_pd_mask(_mm512_loadu_pd(a + i), kv, _CMP_GT_OQ);
        if (m1) return i + __builtin_ctzll((int64_t)m1);
        const __mmask8 m2 = _mm512_cmp_pd_mask(_mm512_loadu_pd(a + i + 8), kv, _CMP_GT_OQ);
        if (m2) return i + 8 + __builtin_ctzll((int64_t)m2);
    }
    for (; i < end; i++)
        if (a[i] > k) return i;
    return -1;
}
#elif defined(__AVX2__)
static inline int64_t scan_chunk(const double *restrict a, int64_t begin, int64_t end,
                                 double k)
{
    const __m256d kv = _mm256_set1_pd(k);
    int64_t i = begin;
    const int64_t iv_end = begin + ((end - begin) & ~(int64_t)7);
    for (; i < iv_end; i += 8) {
        const unsigned m1 = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(a + i), kv,
                                                             _CMP_GT_OQ));
        if (m1) return i + __builtin_ctz(m1);
        const unsigned m2 = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(a + i + 4), kv,
                                                             _CMP_GT_OQ));
        if (m2) return i + 4 + __builtin_ctz(m2);
    }
    for (; i < end; i++)
        if (a[i] > k) return i;
    return -1;
}
#else
static inline int64_t scan_chunk(const double *restrict a, int64_t begin, int64_t end,
                                 double k)
{
    for (int64_t i = begin; i < end; i++)
        if (a[i] > k) return i;
    return -1;
}
#endif

void ext_break_capture_fp64(const double *restrict a,
                            int64_t *restrict out_index,
                            double *restrict out_value,
                            const int64_t K,
                            const int64_t LEN_1D,
                            uint8_t *restrict workspace,
                            const int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;

    const double k = (double)K;
    const int64_t n = LEN_1D;

    if (n <= 0) {
        *out_index = -1;
        *out_value = -1.0;
        return;
    }

    const int nt = omp_get_max_threads();
    if (nt <= 1 || n < (1 << 16)) {
        const int64_t i = scan_chunk(a, 0, n, k);
        if (i < 0) {
            *out_index = -1;
            *out_value = -1.0;
        } else {
            *out_index = i;
            *out_value = a[i];
        }
        return;
    }

    int64_t *found = (int64_t *)malloc(sizeof(int64_t) * (size_t)nt);
    if (found == NULL) {
        const int64_t i = scan_chunk(a, 0, n, k);
        *out_index = i;
        *out_value = (i < 0) ? -1.0 : a[i];
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t t = 0; t < nt; t++) {
        const int64_t s = (n * t) / nt;
        const int64_t e = (n * (t + 1)) / nt;
        found[t] = (e > s) ? scan_chunk(a, s, e, k) : -1;
    }

    int64_t best = -1;
    for (int64_t t = 0; t < nt; t++) {
        const int64_t r = found[t];
        if (r >= 0 && (best < 0 || r < best)) best = r;
    }
    free(found);

    if (best < 0) {
        *out_index = -1;
        *out_value = -1.0;
    } else {
        *out_index = best;
        *out_value = a[best];
    }
}
