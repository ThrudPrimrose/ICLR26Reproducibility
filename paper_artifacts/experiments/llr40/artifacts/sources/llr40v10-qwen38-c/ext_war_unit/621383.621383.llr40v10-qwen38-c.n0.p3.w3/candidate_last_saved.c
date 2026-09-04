/* probe v2: exact-pattern bandwidth probes (XOR sinks, no serial FP chains).
 * probe 1 overwrites b (safe: harness recopies inputs each rep; probe runs on
 * first call = warmup rep, which is discarded). */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

static double clock_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (double)ts.tv_sec*1e9 + ts.tv_nsec; }
static double g_sink[1024];

/* exact kernel traffic: read a[i+1..], read b[i..], ADD, store into OUT (b). 24 B/elem. */
static double probe_kpat(double *restrict a, double *restrict out, int64_t n, int T,
                         int splits, const char *tag)
{
    const int64_t e = n - 1;            /* kernel processes n-1 elements */
    double t0 = clock_ns();
    #pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        __m512i acc = _mm512_setzero_si512();
        const int64_t nit = e;          /* one unit = 8 elements */
        const int64_t step = (nit / T + splits - 1) / splits;
        for (int sp = t; sp < T; sp += T) {   /* sub-range sp (0..T-1) */
            int64_t s = (int64_t)sp * step, ee = s + step;
            if (sp == T - 1) ee = nit;
            if (s >= nit) continue;
            double boundary = a[ee << 3];
            (void)boundary;
            for (int64_t i = s << 3; i + 9 <= ee << 3; i += 8) {
                __m512d va = _mm512_loadu_pd(a + i + 1);
                __m512d vb = _mm512_loadu_pd(out + i);
                __m512d r = _mm512_add_pd(va, vb);
                _mm512_storeu_pd(out + i, r);
                acc = _mm512_xor_si512(acc, _mm512_castpd_si512(r));
            }
            for (int64_t i = (ee<<3) - 7; i < (ee<<3); i++) {
                double v = a[i] + out[i];   /* crude tail, few elems */
                out[i] = v;
                acc = _mm512_xor_epi32(acc, _mm512_castpd_si512(_mm512_set1_pd(v)));
            }
        }
        double d[8]; _mm512_storeu_pd(d, _mm512_castsi512_pd(acc));
        double x = 0; for (int k = 0; k < 8; k++) x += d[k];
        g_sink[t] += x;
    }
    double t1 = clock_ns();
    double s = 0; for (int t = 0; t < T; t++) s += g_sink[t];
    printf("probe %s T=%d split=%d: %.3f ms, %.1f GB/s, sink=%.3e\n",
           tag, T, splits, (t1-t0)*1e-6, (24.0*e)/(t1-t0)/1e9, s);
    return (t1 - t0);
}

/* pure read of a, XOR sink (no FP adds) */
static void probe_read(double *restrict a, int64_t n, int T, const char *tag)
{
    double t0 = clock_ns();
    #pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        __m512i acc = _mm512_setzero_si512();
        int64_t s = (int64_t)t * n / T, e = (int64_t)(t+1) * n / T;
        for (int64_t i = s; i + 8 <= e; i += 8)
            acc = _mm512_xor_si512(acc, _mm512_castpd_si512(_mm512_loadu_pd(a + i)));
        double d[8]; _mm512_storeu_pd(d, _mm512_castsi512_pd(acc));
        double x = 0; for (int k = 0; k < 8; k++) x += d[k];
        g_sink[t] += x;
    }
    double t1 = clock_ns();
    double s = 0; for (int t = 0; t < T; t++) s += g_sink[t];
    printf("probe %s T=%d: %.3f ms, %.1f GB/s, sink=%.3e\n", tag, T, (t1-t0)*1e-6, (8.0*n)/(t1-t0)/1e9, s);
}

static int g_probed = 0;

static void chunk_compute(double *restrict a, const double *restrict b,
                          int64_t s, int64_t e, double boundary)
{
    int64_t i = s;
    for (; i + 9 <= e; i += 8) {
        __m512d vsh = _mm512_loadu_pd(a + i + 1);
        __m5256_1 vb = _mm512_loadu_pd(b + i);
        _mm512_storeu_pd(a + i, _mm512_add_pd(vsh, vb));
    }
    for (; i < e; ++i) {
        double src = (i + 1 < e) ? a[i + 1] : boundary;
        a[i] = src + b[i];
    }
}
