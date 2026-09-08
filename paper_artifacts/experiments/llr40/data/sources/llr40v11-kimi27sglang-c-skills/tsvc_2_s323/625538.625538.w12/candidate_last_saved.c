#include <stdint.h>
#include <x86intrin.h>

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e,
                      const int64_t LEN_1D) {
    if (LEN_1D <= 1) return;
    __m128d prev = _mm_load_sd(&b[0]);
    for (int64_t i = 1; i < LEN_1D; ++i) {
        __m128d ci = _mm_load_sd(&c[i]);
        __m128d cd = _mm_mul_sd(ci, _mm_load_sd(&d[i]));
        __m128d ai = _mm_add_sd(prev, cd);
        _mm_stream_sd(&a[i], ai);
        __m128d ce = _mm_mul_sd(ci, _mm_load_sd(&e[i]));
        prev = _mm_add_sd(ai, ce);
        _mm_store_sd(&b[i], prev);
    }
    _mm_sfence();
}
