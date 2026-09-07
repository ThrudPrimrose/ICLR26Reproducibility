#define _GNU_SOURCE
#include <stdint.h>
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static inline void row512(const double *restrict src, double *restrict dst, int64_t N, int64_t i) {
    const __m512d C = _mm512_set1_pd(0.2);
    const __m512i IC = _mm512_setr_epi32(1,0,2,0,3,0,4,0, 5,0,6,0,7,0,8,0);
    const __m512i IR = _mm512_setr_epi32(2,0,3,0,4,0,5,0, 6,0,7,0,8,0,9,0);
    const double *s = src + i*N;
    const double *su = src + (i+1)*N;
    const double *sd = src + (i-1)*N;
    double *d = dst + i*N;
    d[1] = 0.2 * ((((s[1] + s[0]) + s[2]) + su[1]) + sd[1]);
    int64_t j = 2;
    for (; j <= (int64_t)(N - 23); j += 16) {
        __m512d W1 = _mm512_loadu_pd(s + j - 1);   // s[j-1..j+6]
        __m512d W2 = _mm512_loadu_pd(s + j + 7);   // s[j+7..j+14]
        __m512d W3 = _mm512_loadu_pd(s + j + 15);  // s[j+15..j+22]
        // pass A: outputs d[j..j+7]
        {
            __m512d a = W1;                                  // left
            a = _mm512_add_pd(_mm512_permutex2var_pd(W1, IC, W2), a); // center + left
            a = _mm512_add_pd(a, _mm512_permutex2var_pd(W1, IR, W2)); // + right
            a = _mm512_add_pd(a, _mm512_loadu_pd(su + j));
            a = _mm512_add_pd(a, _mm512_loadu_pd(sd + j));
            _mm512_storeu_pd(d + j, _mm512_mul_pd(a, C));
        }
        // pass B: outputs d[j+8..j+15]
        {
            __m512d b = W2;                                  // left
            b = _mm512_add_pd(_mm512_permutex2var_pd(W2, IC, W3), b);
            b = _mm512_add_pd(b, _mm512_permutex2var_pd(W2, IR, W3));
            b = _mm512_add_pd(b, _mm512_loadu_pd(su + j + 8));
            b = _mm512_add_pd(b, _mm512_loadu_pd(sd + j + 8));
            _mm512_storeu_pd(d + j + 8, _mm512_mul_pd(b, C));
        }
    }
    for (; j < N - 1; ++j)
        d[j] = 0.2 * ((((s[j] + s[j-1]) + s[j+1]) + su[j]) + sd[j]);
}

static int pick_threads(int64_t N) {
    int n = omp_get_max_threads();
    int64_t cap = ((N-2)*(N-2)) / 1369;   // ~16 threads at N=150
    if (cap > n) cap = n;
    if (cap < 1) cap = 1;
    return (int)cap;
}

void jacobi_2d_fp64(double *restrict A, double *restrict B, int64_t N, int64_t TSTEPS) {
    if (N < 4 || TSTEPS <= 0) return;
    int nt = pick_threads(N);
    if (nt <= 1) {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            for (int64_t i = 1; i <= N - 2; ++i) row512(A, B, N, i);
            for (int64_t i = 1; i <= N - 2; ++i) row512(B, A, N, i);
        }
        return;
    }
    omp_set_num_threads(nt);
    #pragma omp parallel
    {
        for (int64_t t = 0; t < TSTEPS; ++t) {
            #pragma omp for schedule(static)
            for (int64_t i = 1; i <= N - 2; ++i) row512(A, B, N, i);
            #pragma omp for schedule(static)
            for (int64_t i = 1; i <= N - 2; ++i) row512(B, A, N, i);
        }
    }
}
