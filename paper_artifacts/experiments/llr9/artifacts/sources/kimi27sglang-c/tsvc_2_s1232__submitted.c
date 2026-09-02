#include <stdint.h>
#include <omp.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                       int64_t LEN_2D, int64_t VLEN) {
    if (LEN_2D <= 0) return;
    #pragma omp parallel for schedule(guided) proc_bind(spread)
    for (int64_t i = 0; i < LEN_2D; ++i) {
        int64_t jmax = i / VLEN;
        const double *restrict brow = bb + i * LEN_2D;
        const double *restrict crow = cc + i * LEN_2D;
        double *restrict arow = aa + i * LEN_2D;
#if defined(__AVX512F__)
        if (jmax >= 32) {
            int64_t j = 0;
            while (((uintptr_t)(arow + j) & 63) != 0) {
                arow[j] = brow[j] + crow[j];
                ++j;
            }
            int64_t jv = j;
            for (; jv + 7 <= jmax; jv += 8) {
                __m512d b = _mm512_loadu_pd(brow + jv);
                __m512d c = _mm512_loadu_pd(crow + jv);
                __m512d s = _mm512_add_pd(b, c);
                _mm512_stream_pd(arow + jv, s);
            }
            for (; jv <= jmax; ++jv) {
                arow[jv] = brow[jv] + crow[jv];
            }
        } else
#elif defined(__AVX2__)
        if (jmax >= 16) {
            int64_t j = 0;
            while (((uintptr_t)(arow + j) & 31) != 0) {
                arow[j] = brow[j] + crow[j];
                ++j;
            }
            int64_t jv = j;
            for (; jv + 3 <= jmax; jv += 4) {
                __m256d b = _mm256_loadu_pd(brow + jv);
                __m256d c = _mm256_loadu_pd(crow + jv);
                __m256d s = _mm256_add_pd(b, c);
                _mm256_stream_pd(arow + jv, s);
            }
            for (; jv <= jmax; ++jv) {
                arow[jv] = brow[jv] + crow[jv];
            }
        } else
#endif
        {
            for (int64_t j = 0; j <= jmax; ++j) {
                arow[j] = brow[j] + crow[j];
            }
        }
    }
}
