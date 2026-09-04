#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

/* Stream compaction: packed[] = src[i]*weight[i] for src[i] > 0, preserving order. */

static int64_t serial_compact(double *restrict packed,
                              const double *restrict src,
                              const double *restrict weight,
                              int64_t n)
{
    int64_t m = 0;
    for (int64_t i = 0; i < n; ++i) {
        double s = src[i];
        if (s > 0.0) {
            packed[m++] = s * weight[i];
        }
    }
    return m;
}

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 const double *restrict src,
                                 const double *restrict weight,
                                 int64_t n,
                                 uint8_t *restrict workspace,
                                 int64_t workspace_bytes)
{
    (void)workspace;
    (void)workspace_bytes;

    if (n <= 0) {
        *out_count = 0;
        return;
    }

    if (n < 4096) {
        *out_count = serial_compact(packed, src, weight, n);
        return;
    }

    const int64_t block = 4096;
    const int64_t nblocks = (n + block - 1) / block;

    int64_t *counts = (int64_t *)malloc((nblocks + 1) * sizeof(int64_t));
    if (!counts) {
        *out_count = serial_compact(packed, src, weight, n);
        return;
    }

    counts[0] = 0;

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblocks; ++b) {
        int64_t lo = b * block;
        int64_t hi = lo + block;
        if (hi > n) hi = n;

        int64_t cnt = 0;
        const __m512d zero = _mm512_setzero_pd();

        int64_t i = lo;
        for (; i + 8 <= hi; i += 8) {
            __m512d v = _mm512_loadu_pd(src + i);
            __mmask8 k = _mm512_cmp_pd_mask(v, zero, _CMP_GT_OQ);
            cnt += (int64_t)_mm_popcnt_u32((unsigned)k);
        }
        for (; i < hi; ++i) {
            if (src[i] > 0.0) ++cnt;
        }
        counts[b + 1] = cnt;
    }

    for (int64_t b = 1; b <= nblocks; ++b) {
        counts[b] += counts[b - 1];
    }

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblocks; ++b) {
        int64_t lo = b * block;
        int64_t hi = lo + block;
        if (hi > n) hi = n;

        double *dst = packed + counts[b];
        const __m512d zero = _mm512_setzero_pd();

        int64_t i = lo;
        for (; i + 16 <= hi; i += 16) {
            __m512d s0 = _mm512_loadu_pd(src + i);
            __m512d w0 = _mm512_loadu_pd(weight + i);
            __m512d s1 = _mm512_loadu_pd(src + i + 8);
            __m512d w1 = _mm512_loadu_pd(weight + i + 8);
            __mmask8 k0 = _mm512_cmp_pd_mask(s0, zero, _CMP_GT_OQ);
            __mmask8 k1 = _mm512_cmp_pd_mask(s1, zero, _CMP_GT_OQ);
            _mm512_mask_compressstoreu_pd(dst, k0, _mm512_mul_pd(s0, w0));
            dst += (int64_t)_mm_popcnt_u32((unsigned)k0);
            _mm512_mask_compressstoreu_pd(dst, k1, _mm512_mul_pd(s1, w1));
            dst += (int64_t)_mm_popcnt_u32((unsigned)k1);
        }
        for (; i + 8 <= hi; i += 8) {
            __m512d s = _mm512_loadu_pd(src + i);
            __m512d w = _mm512_loadu_pd(weight + i);
            __mmask8 k = _mm512_cmp_pd_mask(s, zero, _CMP_GT_OQ);
            _mm512_mask_compressstoreu_pd(dst, k, _mm512_mul_pd(s, w));
            dst += (int64_t)_mm_popcnt_u32((unsigned)k);
        }
        for (; i < hi; ++i) {
            double s = src[i];
            if (s > 0.0) {
                *dst++ = s * weight[i];
            }
        }
    }

    *out_count = counts[nblocks];
    free(counts);
}
