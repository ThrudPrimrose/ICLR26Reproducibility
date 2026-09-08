#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static inline __attribute__((always_inline))
void row_zmm(double *restrict o, const double *restrict a, const double *restrict b, int64_t m) {
    int64_t k = 0;
    while (k < m && ((uintptr_t)(o + k) & 63)) { o[k] = a[k] + b[k]; ++k; }
    for (; k + 96 <= m; k += 96) {
        __m512d v[12];
        #pragma GCC unroll 12
        for (int t = 0; t < 12; ++t) v[t] = _mm512_add_pd(_mm512_loadu_pd(a + k + 8*t), _mm512_loadu_pd(b + k + 8*t));
        #pragma GCC unroll 12
        for (int t = 0; t < 12; ++t) _mm512_storeu_pd(o + k + 8*t, v[t]);
    }
    for (; k < m; ++k) o[k] = a[k] + b[k];
}
void tsvc_2_s119_fp64(double *restrict aa, const double *restrict bb, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    int P = omp_get_max_threads();
    int64_t W = N-1, base = W/P, rem = W%P;
    double *scratch = malloc(8*N*N);
    printf("N=%ld P=%d\n", (long)N, P);
    /* benchD: row-burst, no spin (bandwidth of row-burst pattern) */
    {
        double best = 1e30;
        for (int r = 0; r < 3; ++r) {
            double t0 = now_ns();
            #pragma omp parallel num_threads(P)
            {
                int c = omp_get_thread_num();
                int64_t j0 = 1 + c*base + (c < rem ? c : rem);
                int64_t j1 = 1 + (c+1)*base + (c+1 < rem ? c+1 : rem);
                const double *a = aa + j0;
                const double *b = bb + N + j0;
                double *o = scratch + N + j0;
                for (int64_t i = 1; i < N; ++i) { row_zmm(o, a, b, j1-j0); a += N; b += N; o += N; }
            }
            double dt = now_ns()-t0; if (dt<best) best = dt;
        }
        printf("benchD (row-burst, no spin): %.3f ms  %.1f GB/s\n", best/1e6, (16.0*W*(N-1)/1e9)/(best/1e9));
    }
    /* benchC: long-stream triad */
    {
        double best = 1e30; int64_t tot = 8*N*N/8;
        for (int r = 0; r < 3; ++r) {
            double t0 = now_ns();
            #pragma omp parallel for num_threads(P)
            for (int64_t i = 0; i < tot; i += 8) {
                __m512d x = _mm512_add_pd(_mm512_loadu_pd(aa+i), _mm512_loadu_pd(bb+i));
                _mm512_storeu_pd(scratch+i, x);
            }
            double dt = now_ns()-t0; if (dt<best) best = dt;
        }
        printf("benchC (long-stream triad): %.3f ms  %.1f GB/s\n", best/1e6, (24.0*(8*N*N/8)/1e9)/(best/1e9));
    }
    free(scratch);
    /* real kernel (strip+spin) */
    {
        _Alignas(64) int64_t done[64]; memset(done,0,sizeof done);
        #pragma omp parallel num_threads(P)
        {
            int c = omp_get_thread_num();
            int64_t j0 = 1 + c*base + (c < rem ? c : rem);
            int64_t j1 = 1 + (c+1)*base + (c+1 < rem ? c+1 : rem);
            const double *a = aa + j0 - 1;
            const double *b = bb + N + j0;
            double *o = aa + N + j0;
            for (int64_t i = 1; i < N; ++i) {
                if (c > 0) while (__atomic_load_n(&done[c-1], __ATOMIC_ACQUIRE) < i-1) __builtin_ia32_pause();
                row_zmm(o, a, b, j1-j0);
                __atomic_store_n(&done[c], i, __ATOMIC_RELEASE);
                a += N; b += N; o += N;
            }
        }
    }
    fflush(stdout);
}
