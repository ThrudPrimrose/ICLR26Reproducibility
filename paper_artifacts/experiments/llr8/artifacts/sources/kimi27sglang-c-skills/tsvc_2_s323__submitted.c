#include <stdint.h>
#include <stddef.h>
#include <omp.h>
#include <emmintrin.h>

static inline void stream_sd(double val, double *addr)
{
    __asm__ volatile ("movntsd %1, %0" : "=m" (*addr) : "x" (val));
}

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, double *restrict c,
                      double *restrict d, double *restrict e, int64_t LEN_1D,
                      uint8_t *restrict workspace, int64_t workspace_size)
{
    int64_t n = LEN_1D;
    if (n <= 1) return;

    double bi = b[0];
    for (int64_t i = 1; i < n; i++) {
        _mm_prefetch((const char *)&c[i + 96], _MM_HINT_NTA);
        _mm_prefetch((const char *)&d[i + 96], _MM_HINT_NTA);
        _mm_prefetch((const char *)&e[i + 96], _MM_HINT_NTA);
        double cd = c[i] * d[i];
        double ai = bi + cd;
        double ce = c[i] * e[i];
        bi = ai + ce;
        stream_sd(ai, &a[i]);
        stream_sd(bi, &b[i]);
    }
    __asm__ volatile ("sfence" ::: "memory");
}
