#include <stdint.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void tsvc_2_s311_fp64(const double *restrict a, double *restrict sum_out, const int64_t LEN_1D) {
    const double *restrict p = a;
    if (LEN_1D <= 4096) {
        __m512d s0 = _mm512_setzero_pd();
        __m512d s1 = _mm512_setzero_pd();
        __m512d s2 = _mm512_setzero_pd();
        __m512d s3 = _mm512_setzero_pd();
        __m512d s4 = _mm512_setzero_pd();
        __m512d s5 = _mm512_setzero_pd();
        __m512d s6 = _mm512_setzero_pd();
        __m512d s7 = _mm512_setzero_pd();
        int64_t i = 0;
        for (; i + 64 <= LEN_1D; i += 64) {
            s0 = _mm512_add_pd(s0, _mm512_loadu_pd(p + i));
            s1 = _mm512_add_pd(s1, _mm512_loadu_pd(p + i + 8));
            s2 = _mm512_add_pd(s2, _mm512_loadu_pd(p + i + 16));
            s3 = _mm512_add_pd(s3, _mm512_loadu_pd(p + i + 24));
            s4 = _mm512_add_pd(s4, _mm512_loadu_pd(p + i + 32));
            s5 = _mm512_add_pd(s5, _mm512_loadu_pd(p + i + 40));
            s6 = _mm512_add_pd(s6, _mm512_loadu_pd(p + i + 48));
            s7 = _mm512_add_pd(s7, _mm512_loadu_pd(p + i + 56));
        }
        s0 = _mm512_add_pd(s0, s1);
        s2 = _mm512_add_pd(s2, s3);
        s4 = _mm512_add_pd(s4, s5);
        s6 = _mm512_add_pd(s6, s7);
        s0 = _mm512_add_pd(s0, s2);
        s4 = _mm512_add_pd(s4, s6);
        s0 = _mm512_add_pd(s0, s4);
        double sum = _mm512_reduce_add_pd(s0);
        for (; i < LEN_1D; i++) sum += p[i];
        sum_out[0] = sum;
    } else {
        int nt = omp_get_max_threads();
        __attribute__((aligned(64))) double parts[nt][8];
        #pragma omp parallel num_threads(nt)
        {
            int t = omp_get_thread_num();
            int nt2 = omp_get_num_threads();
            int64_t start = (LEN_1D * t) / nt2;
            int64_t end   = (LEN_1D * (t + 1)) / nt2;
            __m512d s0 = _mm512_setzero_pd();
            __m512d s1 = _mm512_setzero_pd();
            __m512d s2 = _mm512_setzero_pd();
            __m512d s3 = _mm512_setzero_pd();
            __m512d s4 = _mm512_setzero_pd();
            __m512d s5 = _mm512_setzero_pd();
            __m512d s6 = _mm512_setzero_pd();
            __m512d s7 = _mm512_setzero_pd();
            int64_t i = start;
            for (; i + 64 <= end; i += 64) {
                s0 = _mm512_add_pd(s0, _mm512_loadu_pd(p + i));
                s1 = _mm512_add_pd(s1, _mm512_loadu_pd(p + i + 8));
                s2 = _mm512_add_pd(s2, _mm512_loadu_pd(p + i + 16));
                s3 = _mm512_add_pd(s3, _mm512_loadu_pd(p + i + 24));
                s4 = _mm512_add_pd(s4, _mm512_loadu_pd(p + i + 32));
                s5 = _mm512_add_pd(s5, _mm512_loadu_pd(p + i + 40));
                s6 = _mm512_add_pd(s6, _mm512_loadu_pd(p + i + 48));
                s7 = _mm512_add_pd(s7, _mm512_loadu_pd(p + i + 56));
            }
            s0 = _mm512_add_pd(s0, s1);
            s2 = _mm512_add_pd(s2, s3);
            s4 = _mm512_add_pd(s4, s5);
            s6 = _mm512_add_pd(s6, s7);
            s0 = _mm512_add_pd(s0, s2);
            s4 = _mm512_add_pd(s4, s6);
            s0 = _mm512_add_pd(s0, s4);
            double sum = _mm512_reduce_add_pd(s0);
            for (; i < end; i++) sum += p[i];
            parts[t][0] = sum;
        }
        double total = 0.0;
        for (int t = 0; t < nt; t++) total += parts[t][0];
        sum_out[0] = total;
    }
}
