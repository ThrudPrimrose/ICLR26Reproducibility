#include <stdint.h>
#include <immintrin.h>

void fuse_diamond_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    const __m256d one = _mm256_set1_pd(1.0);
    int64_t i = 0;
    while (i < LEN_1D && ((uintptr_t)&a[i] & 31)) {
        double ai = a[i];
        double a2 = ai * ai;
        out[i] = a2 * a2 - 1.0;
        ++i;
    }
    for (; i + 4 <= LEN_1D; i += 4) {
        __m256d va = _mm256_load_pd(&a[i]);
        __m256d v2 = _mm256_mul_pd(va, va);
        _mm256_store_pd(&out[i], _mm256_fmsub_pd(v2, v2, one));
    }
    for (; i < LEN_1D; ++i) {
        double ai = a[i];
        double a2 = ai * ai;
        out[i] = a2 * a2 - 1.0;
    }
}
