#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

static inline void process_chunk(double *restrict a, double *restrict b,
                                 const double *restrict c,
                                 const double *restrict d,
                                 const double *restrict e,
                                 int64_t start, int64_t end) {
    int64_t i = start;

    /* Scalar prefix until b+i is 64-byte aligned for streaming stores. */
    const uintptr_t addr = (uintptr_t)(b + i);
    const int64_t off = (int64_t)((addr & 63u) >> 3);
    const int64_t step = (8 - off) & 7;
    const int64_t pe = (step < (end - i)) ? (i + step) : end;
    for (; i < pe; ++i) {
        const double tmp = d[i] * e[i];
        b[i] = tmp;
        a[i] += tmp * c[i];
    }

    /* Vector body. */
    for (; i + 8 <= end; i += 8) {
        __m512d vd = _mm512_loadu_pd(d + i);
        __m512d ve = _mm512_loadu_pd(e + i);
        __m512d vc = _mm512_loadu_pd(c + i);
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vtmp = _mm512_mul_pd(vd, ve);
        va = _mm512_fmadd_pd(vtmp, vc, va);
        _mm512_stream_pd(b + i, vtmp);
        _mm512_storeu_pd(a + i, va);
    }

    /* Scalar tail. */
    for (; i < end; ++i) {
        const double tmp = d[i] * e[i];
        b[i] = tmp;
        a[i] += tmp * c[i];
    }
}

void tsvc_2_s152_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e,
                      const int64_t LEN_1D) {
    if (LEN_1D <= 0) return;

    /* For small inputs the lightweight combined construct has less overhead. */
    if (LEN_1D < 4096) {
#pragma omp parallel for simd schedule(static)
        for (int64_t i = 0; i < LEN_1D; ++i) {
            const double tmp = d[i] * e[i];
            b[i] = tmp;
            a[i] += tmp * c[i];
        }
        return;
    }

#pragma omp parallel
    {
        const int64_t nt = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();
        const int64_t base = LEN_1D / nt;
        const int64_t rem = LEN_1D % nt;
        const int64_t start = tid * base + (tid < rem ? tid : rem);
        const int64_t end = start + base + (tid < rem ? 1 : 0);

        process_chunk(a, b, c, d, e, start, end);
        _mm_sfence();
    }
}
