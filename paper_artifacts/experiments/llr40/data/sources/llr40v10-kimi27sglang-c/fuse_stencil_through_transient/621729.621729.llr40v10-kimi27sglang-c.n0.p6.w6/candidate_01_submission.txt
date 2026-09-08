#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

static inline void process_block(const double *restrict a, double *restrict out, int64_t i) {
    __m512d v0 = _mm512_loadu_pd(a + i - 1);
    __m512d v1 = _mm512_loadu_pd(a + i);
    __m512d v2 = _mm512_loadu_pd(a + i + 1);
    __m512d v3 = _mm512_loadu_pd(a + i + 2);
    __m512d s = _mm512_add_pd(v1, v2);
    __m512d t0 = _mm512_add_pd(s, v0);
    __m512d t1 = _mm512_add_pd(s, v3);
    __m512d r = _mm512_mul_pd(t0, t1);
    _mm512_storeu_pd(out + i, r);
}

void fuse_stencil_through_transient_fp64(const double *restrict a, double *restrict out, const int64_t LEN_1D) {
    const int64_t start = 1;
    const int64_t end = LEN_1D - 2;

    if (end <= start) return;

    const int64_t n = end - start;
    if (n < 4096) {
        int64_t i = start;
        for (; i + 8 <= end; i += 8) {
            process_block(a, out, i);
        }
        for (; i < end; ++i) {
            double s0 = a[i - 1] + a[i] + a[i + 1];
            double s1 = a[i] + a[i + 1] + a[i + 2];
            out[i] = s0 * s1;
        }
        return;
    }

    const int64_t nblocks = n / 8;
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < nblocks; ++k) {
        process_block(a, out, start + (k << 3));
    }

    int64_t i = start + (nblocks << 3);
    for (; i < end; ++i) {
        double s0 = a[i - 1] + a[i] + a[i + 1];
        double s1 = a[i] + a[i + 1] + a[i + 2];
        out[i] = s0 * s1;
    }
}
