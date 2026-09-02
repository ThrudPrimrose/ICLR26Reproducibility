#include <stdint.h>
#include <immintrin.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    if (LEN_1D < 4096) {
        double sum = 0.0;
        for (int64_t i = 0; i < LEN_1D; ++i) {
            sum += a[i];
            b[i] = sum;
        }
        return;
    }

    double sum = 0.0;
    const int64_t pf = 64;
    for (int64_t i = 0; i < LEN_1D; ++i) {
        if (i + pf < LEN_1D) {
            _mm_prefetch((const char *)&a[i + pf], _MM_HINT_T0);
        }
        sum += a[i];
        __asm__ volatile("movntsd %1, %0" : "=m"(b[i]) : "x"(sum) : "memory");
    }
    _mm_sfence();
}
