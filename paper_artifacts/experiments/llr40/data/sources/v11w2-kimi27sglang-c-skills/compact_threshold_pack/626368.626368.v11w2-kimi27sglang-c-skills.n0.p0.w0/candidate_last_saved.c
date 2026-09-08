#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

static inline int64_t count_chunk(const double *src, int64_t start, int64_t end)
{
    int64_t c = 0;
    int64_t i = start;
    const __m512d zero = _mm512_setzero_pd();
    for (; i + 8 <= end; i += 8) {
        __m512d v = _mm512_loadu_pd(src + i);
        __mmask8 m = _mm512_cmp_pd_mask(v, zero, _CMP_GT_OQ);
        c += (int64_t)__builtin_popcount((unsigned)m);
    }
    for (; i < end; ++i) {
        if (src[i] > 0.0) ++c;
    }
    return c;
}

static inline int64_t write_chunk(const double *src, const double *weight,
                                  double *packed, int64_t start, int64_t end)
{
    int64_t pos = 0;
    int64_t i = start;
    const __m512d zero = _mm512_setzero_pd();
    for (; i + 8 <= end; i += 8) {
        __m512d s = _mm512_loadu_pd(src + i);
        __mmask8 m = _mm512_cmp_pd_mask(s, zero, _CMP_GT_OQ);
        if (m) {
            __m512d w = _mm512_loadu_pd(weight + i);
            __m512d p = _mm512_mul_pd(s, w);
            _mm512_mask_compressstoreu_pd(packed + pos, m, p);
            pos += (int64_t)__builtin_popcount((unsigned)m);
        }
    }
    for (; i < end; ++i) {
        double s = src[i];
        if (s > 0.0) {
            packed[pos++] = s * weight[i];
        }
    }
    return pos;
}

void compact_threshold_pack_fp64(
    int64_t *restrict out_count,
    double *restrict packed,
    const double *restrict src,
    const double *restrict weight,
    int64_t LEN_1D,
    uint8_t *restrict workspace,
    int64_t workspace_size)
{
    (void)workspace;
    (void)workspace_size;

    int64_t n = LEN_1D;
    if (n <= 0) {
        out_count[0] = 0;
        return;
    }

    int nt = omp_get_max_threads();
    if (nt > n) nt = (int)n;
    if (nt < 1) nt = 1;

    if (n < 4096) {
        int64_t pos = 0;
        for (int64_t i = 0; i < n; ++i) {
            double s = src[i];
            if (s > 0.0) {
                packed[pos++] = s * weight[i];
            }
        }
        out_count[0] = pos;
        return;
    }

    int64_t *counts = (int64_t *)malloc((size_t)(nt + 1) * sizeof(int64_t));
    if (counts == NULL) {
        int64_t pos = 0;
        for (int64_t i = 0; i < n; ++i) {
            double s = src[i];
            if (s > 0.0) {
                packed[pos++] = s * weight[i];
            }
        }
        out_count[0] = pos;
        return;
    }

    memset(counts, 0, (size_t)(nt + 1) * sizeof(int64_t));

    #pragma omp parallel num_threads(nt) default(none) shared(src, weight, packed, counts, out_count, n, nt)
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t start = ((int64_t)tid * n) / nthreads;
        int64_t end = ((int64_t)(tid + 1) * n) / nthreads;

        counts[tid] = count_chunk(src, start, end);

        #pragma omp barrier

        #pragma omp single
        {
            int64_t offset = 0;
            for (int i = 0; i < nthreads; ++i) {
                int64_t ci = counts[i];
                counts[i] = offset;
                offset += ci;
            }
            counts[nthreads] = offset;
            out_count[0] = offset;
        }

        #pragma omp barrier

        int64_t pos = counts[tid];
        write_chunk(src + start, weight + start, packed + pos, 0, end - start);
    }

    free(counts);
}
