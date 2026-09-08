#include <stdint.h>
#include <immintrin.h>
#include <omp.h>

void tsvc_2_vtvtv_fp64(double *restrict a, const double *restrict b, const double *restrict c, const int64_t LEN_1D) {
#pragma omp parallel
{
    const int64_t tid = omp_get_thread_num();
    const int64_t nthreads = omp_get_num_threads();
    const int64_t chunk = (LEN_1D + nthreads - 1) / nthreads;
    int64_t start = tid * chunk;
    int64_t end = start + chunk;
    if (end > LEN_1D) end = LEN_1D;

    int64_t aligned_start = (start + 7) & ~((int64_t)7);
    if (aligned_start > end) aligned_start = end;

    int64_t i;
    for (i = start; i < aligned_start; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }

    int64_t main_end = end & ~((int64_t)7);
    for (i = aligned_start; i < main_end; i += 8) {
        __m512d av = _mm512_loadu_pd(a + i);
        __m512d bv = _mm512_loadu_pd(b + i);
        __m512d cv = _mm512_loadu_pd(c + i);
        av = _mm512_mul_pd(av, bv);
        av = _mm512_mul_pd(av, cv);
        _mm512_stream_pd(a + i, av);
    }

    for (i = main_end; i < end; ++i) {
        a[i] = a[i] * b[i] * c[i];
    }
}
    _mm_sfence();
}
