#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void quasi_affine_reduce_odd_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    const int64_t n = LEN_1D / 2;
    const double *restrict p = a;
    const int64_t m = n & ~15;

    double partials[128] = {0.0};

    #pragma omp parallel if(m > 256)
    {
        __m512d v0 = _mm512_setzero_pd();
        __m512d v1 = _mm512_setzero_pd();
        __m512d v2 = _mm512_setzero_pd();
        __m512d v3 = _mm512_setzero_pd();

        #pragma omp for schedule(static) nowait
        for (int64_t i = 0; i < m; i += 16) {
            __m512d x0 = _mm512_loadu_pd(p + 2 * (i + 0));
            __m512d x1 = _mm512_loadu_pd(p + 2 * (i + 4));
            __m512d x2 = _mm512_loadu_pd(p + 2 * (i + 8));
            __m512d x3 = _mm512_loadu_pd(p + 2 * (i + 12));
            v0 = _mm512_add_pd(v0, _mm512_maskz_mov_pd(0xAA, x0));
            v1 = _mm512_add_pd(v1, _mm512_maskz_mov_pd(0xAA, x1));
            v2 = _mm512_add_pd(v2, _mm512_maskz_mov_pd(0xAA, x2));
            v3 = _mm512_add_pd(v3, _mm512_maskz_mov_pd(0xAA, x3));
        }

        __m512d vsum = _mm512_add_pd(_mm512_add_pd(v0, v1), _mm512_add_pd(v2, v3));
        partials[omp_get_thread_num()] = _mm512_reduce_add_pd(vsum);
    }

    double sum = 0.0;
    for (int t = 0; t < omp_get_max_threads(); ++t) {
        sum += partials[t];
    }

    for (int64_t i = m; i < n; ++i) {
        sum += p[2 * i + 1];
    }

    out[0] = sum;
}
