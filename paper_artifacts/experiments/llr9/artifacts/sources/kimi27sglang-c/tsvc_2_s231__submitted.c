#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s231_fp64(double *restrict aa, const double *restrict bb, int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    if (N <= 0) return;

    if (N < 512) {
        for (int64_t j = 1; j < N; ++j) {
            double *restrict arow = aa + j * N;
            const double *restrict aprev = aa + (j - 1) * N;
            const double *restrict brow = bb + j * N;
            int64_t i = 0;
            for (; i + 8 <= N; i += 8) {
                __m512d vprev = _mm512_loadu_pd(aprev + i);
                __m512d vbb = _mm512_loadu_pd(brow + i);
                _mm512_storeu_pd(arow + i, _mm512_add_pd(vprev, vbb));
            }
            for (; i < N; ++i) {
                arow[i] = aprev[i] + brow[i];
            }
        }
        return;
    }

    #pragma omp parallel
    {
        const int64_t nt = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();
        const int64_t i0 = (N * tid) / nt;
        const int64_t i1 = (N * (tid + 1)) / nt;
        const int64_t len = i1 - i0;
        if (len > 0) {
            for (int64_t j = 1; j < N; ++j) {
                const double *restrict aprev = aa + (j - 1) * N + i0;
                const double *restrict brow = bb + j * N + i0;
                double *restrict arow = aa + j * N + i0;
                int64_t i = 0;
                for (; i + 32 <= len; i += 32) {
                    __m512d v0 = _mm512_loadu_pd(aprev + i);
                    __m512d b0 = _mm512_loadu_pd(brow + i);
                    _mm512_storeu_pd(arow + i, _mm512_add_pd(v0, b0));
                    __m512d v1 = _mm512_loadu_pd(aprev + i + 8);
                    __m512d b1 = _mm512_loadu_pd(brow + i + 8);
                    _mm512_storeu_pd(arow + i + 8, _mm512_add_pd(v1, b1));
                    __m512d v2 = _mm512_loadu_pd(aprev + i + 16);
                    __m512d b2 = _mm512_loadu_pd(brow + i + 16);
                    _mm512_storeu_pd(arow + i + 16, _mm512_add_pd(v2, b2));
                    __m512d v3 = _mm512_loadu_pd(aprev + i + 24);
                    __m512d b3 = _mm512_loadu_pd(brow + i + 24);
                    _mm512_storeu_pd(arow + i + 24, _mm512_add_pd(v3, b3));
                }
                for (; i + 8 <= len; i += 8) {
                    __m512d vprev = _mm512_loadu_pd(aprev + i);
                    __m512d vbb = _mm512_loadu_pd(brow + i);
                    _mm512_storeu_pd(arow + i, _mm512_add_pd(vprev, vbb));
                }
                for (; i < len; ++i) {
                    arow[i] = aprev[i] + brow[i];
                }
            }
        }
    }
}
