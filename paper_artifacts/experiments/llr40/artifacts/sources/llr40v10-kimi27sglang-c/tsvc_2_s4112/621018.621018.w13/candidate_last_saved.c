#include <stdint.h>
#include <omp.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

void tsvc_2_s4112_fp64(double *restrict a, const double *restrict b,
                       const int32_t *restrict ip, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

#ifdef __AVX512F__
    const int64_t n = LEN_1D;
    const int64_t VL = 8;
    const int64_t UNR = 4;
    const int64_t chunk = VL * UNR; // 32

    if (n >= chunk) {
        const int64_t limit = n & ~(chunk - 1);
        const __m512d two = _mm512_set1_pd(2.0);

        for (int64_t i = 0; i < limit; i += chunk) {
            // Prefetch index and data for upcoming iterations
            if (i + 512 < limit) {
                _mm_prefetch((const char*)(ip + i + 512), _MM_HINT_T0);
                _mm_prefetch((const char*)(ip + i + 512 + 8), _MM_HINT_T0);
                _mm_prefetch((const char*)(ip + i + 512 + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(ip + i + 512 + 24), _MM_HINT_T0);
            }

            __m256i idx0 = _mm256_loadu_si256((const __m256i*)(ip + i + 0 * VL));
            __m256i idx1 = _mm256_loadu_si256((const __m256i*)(ip + i + 1 * VL));
            __m256i idx2 = _mm256_loadu_si256((const __m256i*)(ip + i + 2 * VL));
            __m256i idx3 = _mm256_loadu_si256((const __m256i*)(ip + i + 3 * VL));

            // Prefetch b values that will be gathered a little later
            if (i + 256 < limit) {
                _mm_prefetch((const char*)&b[ip[i + 256 + 0]], _MM_HINT_T0);
                _mm_prefetch((const char*)&b[ip[i + 256 + 1]], _MM_HINT_T0);
                _mm_prefetch((const char*)&b[ip[i + 256 + 2]], _MM_HINT_T0);
                _mm_prefetch((const char*)&b[ip[i + 256 + 3]], _MM_HINT_T0);
            }

            __m512d vb0 = _mm512_i32gather_pd(idx0, b, 8);
            __m512d vb1 = _mm512_i32gather_pd(idx1, b, 8);
            __m512d vb2 = _mm512_i32gather_pd(idx2, b, 8);
            __m512d vb3 = _mm512_i32gather_pd(idx3, b, 8);

            __m512d va0 = _mm512_loadu_pd(a + i + 0 * VL);
            __m512d va1 = _mm512_loadu_pd(a + i + 1 * VL);
            __m512d va2 = _mm512_loadu_pd(a + i + 2 * VL);
            __m512d va3 = _mm512_loadu_pd(a + i + 3 * VL);

            va0 = _mm512_fmadd_pd(vb0, two, va0);
            va1 = _mm512_fmadd_pd(vb1, two, va1);
            va2 = _mm512_fmadd_pd(vb2, two, va2);
            va3 = _mm512_fmadd_pd(vb3, two, va3);

            _mm512_storeu_pd(a + i + 0 * VL, va0);
            _mm512_storeu_pd(a + i + 1 * VL, va1);
            _mm512_storeu_pd(a + i + 2 * VL, va2);
            _mm512_storeu_pd(a + i + 3 * VL, va3);
        }
        for (int64_t i = limit; i < n; ++i) {
            a[i] += b[ip[i]] * 2.0;
        }
        return;
    }
#endif

    for (int64_t i = 0; i < LEN_1D; ++i) {
        a[i] += b[ip[i]] * 2.0;
    }
}
