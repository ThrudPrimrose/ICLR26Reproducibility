#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

static inline double hsum8(__m512d v)
{
    return _mm512_reduce_add_pd(v);
}

void tsvc_2_s311_fp64(double *restrict a, double *restrict sum_out,
                      int64_t LEN_1D, unsigned char *restrict workspace,
                      int64_t workspace_size)
{
    (void)workspace; (void)workspace_size;
    if (LEN_1D <= 0) { sum_out[0] = 0.0; return; }
    double total = 0.0;
    #pragma omp parallel reduction(+:total)
    {
        const int    nt  = omp_get_num_threads();
        const int    tid = omp_get_thread_num();
        const int64_t chunk = (LEN_1D + nt - 1) / nt;
        int64_t b0 = (int64_t)tid * chunk;
        int64_t b1 = b0 + chunk; if (b1 > LEN_1D) b1 = LEN_1D;
        const double *restrict p = a + b0;
        const int64_t n = b1 - b0;

        __m512d s0 = _mm512_setzero_pd(), s1 = _mm512_setzero_pd(),
                s2 = _mm512_setzero_pd(), s3 = _mm512_setzero_pd(),
                s4 = _mm512_setzero_pd(), s5 = _mm512_setzero_pd(),
                s6 = _mm512_setzero_pd(), s7 = _mm512_setzero_pd();
        int64_t i = 0;
        const int64_t n64 = n & ~(int64_t)63;
        for (; i < n64; i += 64) {
            s0 = _mm512_add_pd(s0, _mm512_loadu_pd(p + i));
            s1 = _mm512_add_pd(s1, _mm512_loadu_pd(p + i +  8));
            s2 = _mm512_add_pd(s2, _mm512_loadu_pd(p + i + 16));
            s3 = _mm512_add_pd(s3, _mm512_loadu_pd(p + i + 24));
            s4 = _mm512_add_pd(s4, _mm512_loadu_pd(p + i + 32));
            s5 = _mm512_add_pd(s5, _mm512_loadu_pd(p + i + 40));
            s6 = _mm512_add_pd(s6, _mm512_loadu_pd(p + i + 48));
            s7 = _mm512_add_pd(s7, _mm512_loadu_pd(p + i + 56));
        }
        double t = hsum8(s0) + hsum8(s1) + hsum8(s2) + hsum8(s3)
                 + hsum8(s4) + hsum8(s5) + hsum8(s6) + hsum8(s7);
        for (; i < n; ++i) t += p[i];
        total += t;
    }
    sum_out[0] = total;
}
