/* Stream compaction: packed[0..n) = src[i]*weight[i] for src[i] > 0, in source order.
 * 3-phase: parallel per-chunk counts, serial scan of chunk counts, parallel compress-store pack. */
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

void compact_threshold_pack_fp64(int64_t *restrict out_count,
                                 double *restrict packed,
                                 const double *restrict src,
                                 const double *restrict weight,
                                 const int64_t LEN_1D,
                                 uint8_t *restrict workspace,
                                 const int64_t workspace_bytes)
{
    if (LEN_1D <= 0) {
        out_count[0] = 0;
        return;
    }

    /* Small inputs: stay serial, avoid the fork cost. */
    if (LEN_1D < (1 << 18)) {
        int64_t n = 0;
        for (int64_t i = 0; i < LEN_1D; i++)
            if (src[i] > 0.0)
                packed[n++] = src[i] * weight[i];
        out_count[0] = n;
        return;
    }

    const int nt = omp_get_max_threads();
    int64_t nch = (int64_t)nt * 32;
    if (nch < 16) nch = 16;
    if (nch > 8192) nch = 8192;
    int64_t chunk = (LEN_1D + nch - 1) / nch;
    chunk = (chunk + 7) & ~7; /* multiple of 8 doubles */
    nch = (LEN_1D + chunk - 1) / chunk;

    int64_t *off;
    const int have_ws = (workspace_bytes >= (nch + 1) * 8);
    if (have_ws)
        off = (int64_t *)(void *)workspace;
    else
        off = (int64_t *)malloc((nch + 1) * sizeof(int64_t));

#pragma omp parallel
    {
        /* Phase 1: survivors per chunk. Reads src only. */
#pragma omp for schedule(static)
        for (int64_t c = 0; c < nch; c++) {
            const int64_t lo = c * chunk;
            int64_t rem = lo + chunk > LEN_1D ? LEN_1D - lo : chunk;
            const double *s = src + lo;
            const __m512d zero = _mm512_setzero_pd();
            int64_t cnt = 0;
            int64_t i = 0;
            for (; i + 16 <= rem; i += 16) {
                const __m512d a = _mm512_loadu_pd(s + i);
                const __m512d b = _mm512_loadu_pd(s + i + 8);
                const __mmask8 ma = _mm512_cmp_pd_mask(a, zero, _CMP_GT_OQ);
                const __mmask8 mb = _mm512_cmp_pd_mask(b, zero, _CMP_GT_OQ);
                /* NOTE: no direct mask->int64 casts: GCC 16 trunk miscompiles them. */
                cnt += (int64_t)(_mm_popcnt_u32(ma) + _mm_popcnt_u32(mb));
            }
            for (; i < rem; i++)
                cnt += (s[i] > 0.0);
            off[c + 1] = cnt;
        }

        /* Phase 2: exclusive scan of chunk counts (tiny, serial). */
#pragma omp single
        {
            int64_t acc = 0;
            for (int64_t c = 0; c < nch; c++) {
                const int64_t t = off[c + 1];
                off[c] = acc;
                acc += t;
            }
            out_count[0] = acc;
        }

        /* Phase 3: compacted pack with AVX-512 compress stores. */
#pragma omp for schedule(static)
        for (int64_t c = 0; c < nch; c++) {
            const int64_t lo = c * chunk;
            int64_t rem = lo + chunk > LEN_1D ? LEN_1D - lo : chunk;
            const double *s = src + lo;
            const double *w = weight + lo;
            double *d = packed + off[c];
            const __m512d zero = _mm512_setzero_pd();
            int64_t i = 0, cur = 0;
            for (; i + 16 <= rem; i += 16) {
                const __m512d v0 = _mm512_loadu_pd(s + i);
                const __m512d v1 = _mm512_loadu_pd(s + i + 8);
                const __mmask8 m0 = _mm512_cmp_pd_mask(v0, zero, _CMP_GT_OQ);
                const __mmask8 m1 = _mm512_cmp_pd_mask(v1, zero, _CMP_GT_OQ);
                if (m0) {
                    const __m512d pw = _mm512_loadu_pd(w + i);
                    const __m512d pr = _mm512_mul_pd(v0, pw);
                    _mm512_mask_compressstoreu_pd(d + cur, m0, pr);
                    cur += _mm_popcnt_u32(m0);
                }
                if (m1) {
                    const __m512d pw = _mm512_loadu_pd(w + i + 8);
                    const __m512d pr = _mm512_mul_pd(v1, pw);
                    _mm512_mask_compressstoreu_pd(d + cur, m1, pr);
                    cur += _mm_popcnt_u32(m1);
                }
            }
            for (; i + 8 <= rem; i += 8) {
                const __m512d v = _mm512_loadu_pd(s + i);
                const __mmask8 m = _mm512_cmp_pd_mask(v, zero, _CMP_GT_OQ);
                if (m) {
                    const __m512d pw = _mm512_loadu_pd(w + i);
                    _mm512_mask_compressstoreu_pd(d + cur, m, _mm512_mul_pd(v, pw));
                    cur += _mm_popcnt_u32(m);
                }
            }
            for (; i < rem; i++)
                if (s[i] > 0.0)
                    d[cur++] = s[i] * w[i];
        }
    }

    if (!have_ws)
        free(off);
}
