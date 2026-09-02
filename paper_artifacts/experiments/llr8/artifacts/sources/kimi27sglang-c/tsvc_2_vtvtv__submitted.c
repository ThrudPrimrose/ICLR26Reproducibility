#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

#define PARALLEL_THRESHOLD 4096

static void __attribute__((noinline)) large_vtvtv(double *restrict a, const double *restrict b, const double *restrict c, int64_t LEN_1D) {
    int64_t i = 0;
    while (i < LEN_1D &&
           (((uintptr_t)(a + i) | (uintptr_t)(b + i) | (uintptr_t)(c + i)) & 63ULL)) {
        a[i] = a[i] * b[i] * c[i];
        ++i;
    }
    int64_t n = LEN_1D - i;
    int64_t m8 = n & ~(int64_t)7;

    if (m8 > 0) {
#pragma omp parallel for schedule(static)
        for (int64_t k = 0; k < m8; k += 8) {
            __m512d va = _mm512_load_pd(a + i + k);
            __m512d vb = _mm512_load_pd(b + i + k);
            __m512d vc = _mm512_load_pd(c + i + k);
            _mm512_store_pd(a + i + k, _mm512_mul_pd(va, _mm512_mul_pd(vb, vc)));
        }
        i += m8;
    }

    for (; i < LEN_1D; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }
}

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
    if (LEN_1D < PARALLEL_THRESHOLD) {
        for (int64_t i = 0; i < LEN_1D; ++i) {
            a[i] = a[i] * b[i] * c[i];
        }
        return;
    }
    large_vtvtv(a, b, c, LEN_1D);
}
