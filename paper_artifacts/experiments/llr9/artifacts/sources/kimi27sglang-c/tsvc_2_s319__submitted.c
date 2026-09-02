#include <stdint.h>
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    if (n <= 0) {
        b[0] = 0.0;
        return;
    }

    const int64_t n8 = n & ~((int64_t)7);
    double s = 0.0;

    const int max_threads = omp_get_max_threads();
    int nthr;
    if (n < 4096) {
        nthr = 1;
    } else {
        nthr = (int)((n + 16383) / 16384);
        if (nthr < 4) nthr = 4;
        if (nthr > max_threads) nthr = max_threads;
        if (nthr > 32) nthr = 32;          /* best trade-off on the target host */
    }

    const int use_stream = (n >= (1 << 20)) &&
                           (((uintptr_t)a & 63) == 0) &&
                           (((uintptr_t)b & 63) == 0);

    if (nthr == 1) {
        __m512d vsum_a = _mm512_setzero_pd();
        __m512d vsum_b = _mm512_setzero_pd();
        int64_t i = 0;
        if (use_stream) {
            for (; i < n8; i += 8) {
                __m512d vc = _mm512_loadu_pd(&c[i]);
                __m512d vd = _mm512_loadu_pd(&d[i]);
                __m512d ve = _mm512_loadu_pd(&e[i]);
                __m512d va = _mm512_add_pd(vc, vd);
                __m512d vb = _mm512_add_pd(vc, ve);
                _mm512_stream_pd(&a[i], va);
                _mm512_stream_pd(&b[i], vb);
                vsum_a = _mm512_add_pd(vsum_a, va);
                vsum_b = _mm512_add_pd(vsum_b, vb);
            }
            _mm_sfence();
        } else {
            for (; i < n8; i += 8) {
                __m512d vc = _mm512_loadu_pd(&c[i]);
                __m512d vd = _mm512_loadu_pd(&d[i]);
                __m512d ve = _mm512_loadu_pd(&e[i]);
                __m512d va = _mm512_add_pd(vc, vd);
                __m512d vb = _mm512_add_pd(vc, ve);
                _mm512_storeu_pd(&a[i], va);
                _mm512_storeu_pd(&b[i], vb);
                vsum_a = _mm512_add_pd(vsum_a, va);
                vsum_b = _mm512_add_pd(vsum_b, vb);
            }
        }
        s = _mm512_reduce_add_pd(vsum_a) + _mm512_reduce_add_pd(vsum_b);
        for (; i < n; ++i) {
            a[i] = c[i] + d[i];
            s += a[i];
            b[i] = c[i] + e[i];
            s += b[i];
        }
        b[0] = s;
        return;
    }

    double partial[32];
    for (int t = 0; t < nthr; ++t) partial[t] = 0.0;

    #pragma omp parallel num_threads(nthr)
    {
        const int tid = omp_get_thread_num();
        __m512d vsum_a = _mm512_setzero_pd();
        __m512d vsum_b = _mm512_setzero_pd();

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n8; i += 8) {
            __m512d vc = _mm512_loadu_pd(&c[i]);
            __m512d vd = _mm512_loadu_pd(&d[i]);
            __m512d ve = _mm512_loadu_pd(&e[i]);
            __m512d va = _mm512_add_pd(vc, vd);
            __m512d vb = _mm512_add_pd(vc, ve);
            if (use_stream) {
                _mm512_stream_pd(&a[i], va);
                _mm512_stream_pd(&b[i], vb);
            } else {
                _mm512_storeu_pd(&a[i], va);
                _mm512_storeu_pd(&b[i], vb);
            }
            vsum_a = _mm512_add_pd(vsum_a, va);
            vsum_b = _mm512_add_pd(vsum_b, vb);
        }
        partial[tid] = _mm512_reduce_add_pd(vsum_a) + _mm512_reduce_add_pd(vsum_b);
    }

    if (use_stream) _mm_sfence();

    for (int t = 0; t < nthr; ++t) s += partial[t];

    for (int64_t i = n8; i < n; ++i) {
        a[i] = c[i] + d[i];
        s += a[i];
        b[i] = c[i] + e[i];
        s += b[i];
    }

    b[0] = s;
}
