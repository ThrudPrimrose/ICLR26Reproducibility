#include <stdint.h>
#include <omp.h>
#include <stdio.h>
#include <immintrin.h>

static const __m512i IDX15 = {15,14,13,12,11,10,9,8};

void versioned_distance_update_fp64(double *restrict a, double *restrict b, double *restrict c,
                                    const int64_t K, const int64_t LEN_1D,
                                    uint8_t *workspace, const int64_t workspace_size) {
    (void)K; (void)workspace; (void)workspace_size;
    int64_t n = LEN_1D;
    double acc = 0;
    __m512d one75 = _mm512_set1_pd(0.75);

    /* 1. read BW: sum b + c in parallel */
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(6) reduction(+:acc)
    {
        double s = 0;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; i += 8) {
            __m512d v = _mm512_add_pd(_mm512_loadu_pd(b + i), _mm512_loadu_pd(c + i));
            s += _mm512_reduce_add_pd(v);
        }
        acc += s;
    }
    double t1 = omp_get_wtime();
    printf("read_BW(2 arrays)=%.1f GB/s\n", (2.0 * n * 8) / (t1 - t0) / 1e9);

    /* 2a. plain-store RW BW: a[i] = a[i]+1 */
    t0 = omp_get_wtime();
    #pragma omp parallel num_threads(6)
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < n; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        _mm512_storeu_pd(a + i, _mm512_add_pd(va, _mm512_set1_pd(1.0)));
    }
    double t2 = omp_get_wtime();
    printf("rw_BW(plain)=%.1f GB/s\n", (2.0 * n * 8) / (t2 - t1) / 1e9);

    /* 2b. NT-store W BW: a[i] = i-ish */
    t0 = omp_get_wtime();
    #pragma omp parallel num_threads(6)
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < n; i += 8)
        _mm512_stream_pd(a + i, _mm512_set1_pd((double)i));
    double t3 = omp_get_wtime();
    printf("w_BW(NT)=%.1f GB/s\n", (1.0 * n * 8) / (t3 - t2) / 1e9);

    /* 3. 1-thread SIMD scheme-B chain, n elements: x=0.75x+t, t=b[i]*c[i] */
    t0 = omp_get_wtime();
    {
        double x = 1.0;
        int64_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512d vt = _mm512_mul_pd(_mm512_loadu_pd(b + i), _mm512_loadu_pd(c + i));
            __m512d vx = _mm512_fmadd_pd(one75, _mm512_set1_pd(x), vt);
            #pragma GCC unroll 15
            for (int s = 0; s < 15; s++)
                vx = _mm512_fmadd_pd(one75, _mm512_permutevar_pd(vx, IDX15), vt);
            { double _t[16]; _mm512_storeu_pd(_t, vx); x = _t[15]; }
        }
        for (; i < n; i++) x = 0.75 * x + b[i] * c[i];
        a[0] = x;
    }
    double t4 = omp_get_wtime();
    printf("chain1t: %.2f cycles/elem @3.6GHz, %d c/elem @1.8\n",
           (t4 - t3) * 3.6e9 / n, (int)((t4 - t3) * 3.6e9 / n * 2));
    if (acc == 12345.0) printf("acc=%.3f\n", acc);
    fflush(stdout);
}
