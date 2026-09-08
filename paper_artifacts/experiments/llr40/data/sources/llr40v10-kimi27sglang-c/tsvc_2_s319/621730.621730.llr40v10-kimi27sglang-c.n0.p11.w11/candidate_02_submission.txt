#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e,
                      const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    if (n <= 0) {
        b[0] = 0.0;
        return;
    }

    double sum = 0.0;

    const int aligned_out = (((uintptr_t)a & 63) == 0) && (((uintptr_t)b & 63) == 0);
    const int64_t nvec = n & ~15;          // multiple of 16

    if (nvec > 0 && aligned_out) {
        #pragma omp parallel
        {
            __m512d acc0 = _mm512_setzero_pd();
            __m512d acc1 = _mm512_setzero_pd();

            #pragma omp for nowait schedule(static)
            for (int64_t i = 0; i < nvec; i += 16) {
                __m512d c0 = _mm512_loadu_pd(&c[i]);
                __m512d c1 = _mm512_loadu_pd(&c[i + 8]);
                __m512d d0 = _mm512_loadu_pd(&d[i]);
                __m512d d1 = _mm512_loadu_pd(&d[i + 8]);
                __m512d e0 = _mm512_loadu_pd(&e[i]);
                __m512d e1 = _mm512_loadu_pd(&e[i + 8]);

                __m512d a0 = _mm512_add_pd(c0, d0);
                __m512d a1 = _mm512_add_pd(c1, d1);
                __m512d b0 = _mm512_add_pd(c0, e0);
                __m512d b1 = _mm512_add_pd(c1, e1);

                _mm512_stream_pd(&a[i],     a0);
                _mm512_stream_pd(&a[i + 8], a1);
                _mm512_stream_pd(&b[i],     b0);
                _mm512_stream_pd(&b[i + 8], b1);

                acc0 = _mm512_add_pd(acc0, _mm512_add_pd(a0, b0));
                acc1 = _mm512_add_pd(acc1, _mm512_add_pd(a1, b1));
            }

            double local = _mm512_reduce_add_pd(acc0) + _mm512_reduce_add_pd(acc1);
            #pragma omp atomic
            sum += local;
        }
        _mm_sfence();
    } else if (nvec > 0) {
        /* Aligned vector path without non-temporal stores (pointers not 64-byte aligned). */
        #pragma omp parallel
        {
            __m512d acc0 = _mm512_setzero_pd();
            __m512d acc1 = _mm512_setzero_pd();

            #pragma omp for nowait schedule(static)
            for (int64_t i = 0; i < nvec; i += 16) {
                __m512d c0 = _mm512_loadu_pd(&c[i]);
                __m512d c1 = _mm512_loadu_pd(&c[i + 8]);
                __m512d d0 = _mm512_loadu_pd(&d[i]);
                __m512d d1 = _mm512_loadu_pd(&d[i + 8]);
                __m512d e0 = _mm512_loadu_pd(&e[i]);
                __m512d e1 = _mm512_loadu_pd(&e[i + 8]);

                __m512d a0 = _mm512_add_pd(c0, d0);
                __m512d a1 = _mm512_add_pd(c1, d1);
                __m512d b0 = _mm512_add_pd(c0, e0);
                __m512d b1 = _mm512_add_pd(c1, e1);

                _mm512_storeu_pd(&a[i],     a0);
                _mm512_storeu_pd(&a[i + 8], a1);
                _mm512_storeu_pd(&b[i],     b0);
                _mm512_storeu_pd(&b[i + 8], b1);

                acc0 = _mm512_add_pd(acc0, _mm512_add_pd(a0, b0));
                acc1 = _mm512_add_pd(acc1, _mm512_add_pd(a1, b1));
            }

            double local = _mm512_reduce_add_pd(acc0) + _mm512_reduce_add_pd(acc1);
            #pragma omp atomic
            sum += local;
        }
    }

    /* Scalar tail (also handles very small n). */
    for (int64_t i = nvec; i < n; ++i) {
        double ai = c[i] + d[i];
        double bi = c[i] + e[i];
        a[i] = ai;
        b[i] = bi;
        sum += ai + bi;
    }

    b[0] = sum;
}
