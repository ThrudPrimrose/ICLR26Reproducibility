#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

/* TSVC s311: sum_out[0] = sum_{i=0}^{LEN_1D-1} a[i]   (fp64)
 * N-thread OpenMP + vectorized per-thread partial sums (AVX-512 / AVX2 / serial).
 */

static double sum_serial(const double *p, int64_t n)
{
    double s0=0,s1=0,s2=0,s3=0,s4=0,s5=0,s6=0,s7=0;
    int64_t i = 0;
    int64_t n8 = n & ~(int64_t)7;
    for (; i < n8; i += 8) {
        s0 += p[i+0]; s1 += p[i+1]; s2 += p[i+2]; s3 += p[i+3];
        s4 += p[i+4]; s5 += p[i+5]; s6 += p[i+6]; s7 += p[i+7];
    }
    double s = s0+s4, t = s1+s5, u = s2+s6, v = s3+s7;
    double r = (s+u) + (t+v);
    for (; i < n; i++) r += p[i];
    return r;
}

__attribute__((target("avx2")))
static double hsum256(__m256d v)
{
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s = _mm_add_pd(lo, hi);
    s = _mm_add_pd(s, _mm_unpackhi_pd(s, s));
    return _mm_cvtsd_f64(s);
}

__attribute__((target("avx2")))
static double sum_avx2(const double *p, int64_t n)
{
    __m256d v0=_mm256_setzero_pd(), v1=_mm256_setzero_pd(),
            v2=_mm256_setzero_pd(), v3=_mm256_setzero_pd();
    int64_t i = 0;
    int64_t n16 = n & ~(int64_t)15;
    for (; i < n16; i += 16) {
        v0 = _mm256_add_pd(v0, _mm256_loadu_pd(p+i));
        v1 = _mm256_add_pd(v1, _mm256_loadu_pd(p+i+4));
        v2 = _mm256_add_pd(v2, _mm256_loadu_pd(p+i+8));
        v3 = _mm256_add_pd(v3, _mm256_loadu_pd(p+i+12));
    }
    double r = hsum256(v0+v2) + hsum256(v1+v3);
    for (; i < n; i++) r += p[i];
    return r;
}

__attribute__((target("avx512f")))
static double sum_avx512(const double *p, int64_t n)
{
    __m512d v0=_mm512_setzero_pd(), v1=_mm512_setzero_pd(),
            v2=_mm512_setzero_pd(), v3=_mm512_setzero_pd();
    int64_t i = 0;
    int64_t n32 = n & ~(int64_t)31;
    for (; i < n32; i += 32) {
        v0 = _mm512_add_pd(v0, _mm512_loadu_pd(p+i));
        v1 = _mm512_add_pd(v1, _mm512_loadu_pd(p+i+8));
        v2 = _mm512_add_pd(v2, _mm512_loadu_pd(p+i+16));
        v3 = _mm512_add_pd(v3, _mm512_loadu_pd(p+i+24));
    }
    double tmp[32];
    _mm512_storeu_pd(tmp + 0, v0);
    _mm512_storeu_pd(tmp + 8, v1);
    _mm512_storeu_pd(tmp + 16, v2);
    _mm512_storeu_pd(tmp + 24, v3);
    double r = 0.0;
    for (int k = 0; k < 32; k++) r += tmp[k];
    for (; i < n; i++) r += p[i];
    return r;
}

void tsvc_2_s311_fp64(double *restrict a, double *restrict sum_out, int64_t LEN_1D)
{
    int64_t N = LEN_1D;
    if (N <= 0) return;
    int P = omp_get_max_threads();
    if (P < 1) P = 1;
    if ((int64_t)P > N / 4096 + 1) P = (int)(N / 4096 + 1);
    if (P < 1) P = 1;

    const int use512 = __builtin_cpu_supports("avx512f");
    const int use256 = !use512 && __builtin_cpu_supports("avx2");

    double partial[2048];
    #pragma omp parallel num_threads(P)
    {
        int tid = omp_get_thread_num();
        int64_t n = (N + P - 1) / P;
        int64_t b0 = (int64_t)tid * n;
        int64_t b1 = b0 + n;
        if (b1 > N) b1 = N;
        const double *p = a + b0;
        int64_t m = b1 - b0;
        double s;
        if (use512) s = sum_avx512(p, m);
        else if (use256) s = sum_avx2(p, m);
        else s = sum_serial(p, m);
        partial[tid] = s;
    }
    /* pairwise combine */
    double acc[2048];
    for (int i = 0; i < P; i++) acc[i] = partial[i];
    int cnt = P;
    while (cnt > 1) {
        for (int i = 0; i + 1 < cnt; i += 2) acc[i >> 1] = acc[i] + acc[i + 1];
        if (cnt & 1) acc[cnt >> 1] = acc[cnt - 1];
        cnt = (cnt + 1) >> 1;
    }
    *sum_out = acc[0];
}
