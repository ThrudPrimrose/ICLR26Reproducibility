#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

static double b0glob;
static const double *cg, *dg, *eg;
static double *ag, *bg;
static int64_t NG;
static double *CS;   /* chunk sums */

static void chunk_bounds(int tid, int nT, int64_t *lo, int64_t *hi) {
    int64_t m = NG - 1;
    int64_t base = m / nT;
    int64_t rem = m - base * nT;
    *lo = 1 + base*(int64_t)tid + (tid < rem ? tid : rem);
    *hi = 1 + base*(int64_t)(tid+1) + ((tid+1) < rem ? tid+1 : rem);
}

/* v1 pass2-style: double rounding per element */
static void run_v1(int64_t n) {
    double b0 = bg[0];
#pragma omp parallel
    {
        int nT = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t lo, hi; chunk_bounds(tid, nT, &lo, &hi);
        double acc = 0.0;
        for (int64_t i = lo; i < hi; ++i) acc += cg[i]*dg[i] + cg[i]*eg[i];
        CS[tid] = acc;
#pragma omp barrier
        double off = b0;
        for (int j = 0; j < tid; ++j) off += CS[j];
        double prev = off;
        for (int64_t i = lo; i < hi; ++i) {
            double u = cg[i]*dg[i];
            ag[i] = prev + u;
            bg[i] = ag[i] + cg[i]*eg[i];
            prev = bg[i];
        }
    }
}

/* v5: double chunksum, bitwise-fma per element */
static void run_v5(int64_t n) {
    double b0 = bg[0];
#pragma omp parallel
    {
        int nT = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t lo, hi; chunk_bounds(tid, nT, &lo, &hi);
        double acc = 0.0;
        for (int64_t i = lo; i < hi; ++i) acc += fma(cg[i], eg[i], cg[i]*dg[i]);
        CS[tid] = acc;
#pragma omp barrier
        double off = b0;
        for (int j = 0; j < tid; ++j) off += CS[j];
        double prev = off;
        for (int64_t i = lo; i < hi; ++i) {
            ag[i] = fma(cg[i], dg[i], prev);
            bg[i] = fma(cg[i], eg[i], ag[i]);
            prev = bg[i];
        }
    }
}

/* v6: long double (80-bit) chunksum, bitwise-fma per element */
static void run_v6(int64_t n, long double *CSld) {
    double b0 = bg[0];
#pragma omp parallel
    {
        int nT = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int64_t lo, hi; chunk_bounds(tid, nT, &lo, &hi);
        long double acc = 0.0L;
        for (int64_t i = lo; i < hi; ++i)
            acc += (long double)cg[i]*dg[i] + (long double)cg[i]*eg[i];
        CSld[tid] = acc;
#pragma omp barrier
        long double off = (long double)b0;
        for (int j = 0; j < tid; ++j) off += CSld[j];
        double prev = (double)off;
        for (int64_t i = lo; i < hi; ++i) {
            ag[i] = fma(cg[i], dg[i], prev);
            bg[i] = fma(cg[i], eg[i], ag[i]);
            prev = bg[i];
        }
    }
}

static void report(const char *name, const double *ra, const double *rb, int64_t n) {
    double mr1 = 0, mr2 = 0, md = 0;
    int64_t w1 = -1, w2 = -1, wd = -1;
    for (int64_t i = 1; i < n; ++i) {
        double da = fabs(ag[i]-ra[i]), db = fabs(bg[i]-rb[i]);
        double d = da > db ? da : db;
        double r = fmax(fabs(ra[i]), fabs(rb[i]));
        if (d > md) { md = d; wd = i; }
        double b1 = 1.57e-7 * r;               /* atol=0 */
        double b2 = 1.1145e-5;                 /* rtol=0 */
        double q1 = d / b1, q2 = d / b2;
        if (q1 > mr1) { mr1 = q1; w1 = i; }
        if (q2 > mr2) { mr2 = q2; w2 = i; }
    }
    printf("%s: maxD=%.3e@%ld | ratio_rtol=%.2f@%ld (|ref|=%.4g) | ratio_atol=%.2f@%ld (|ref|=%.4g)\n",
           name, md, (long)wd, mr1, (long)w1, (w1>0?fabs(rb[w1]):-1), mr2, (long)w2, (w2>0?fabs(rb[w2]):-1));
}

void tsvc_2_s323_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e, const int64_t LEN_1D) {
    int64_t n = LEN_1D;
    int T = omp_get_max_threads();
    int64_t m = n - 1;
    double *ra = malloc(n*8), *rb = malloc(n*8);
    for (int64_t i = 0; i < n; ++i) { ra[i] = a[i]; rb[i] = b[i]; }
    /* reference 1: fma-contracted */
    for (int64_t i = 1; i < n; ++i) { ra[i] = fma(c[i], d[i], rb[i-1]); rb[i] = fma(c[i], e[i], ra[i]); }
    /* reference 2: no contraction */
    double *ra2 = malloc(n*8), *rb2 = malloc(n*8);
    for (int64_t i = 0; i < n; ++i) { ra2[i] = a[i]; rb2[i] = b[i]; }
    for (int64_t i = 1; i < n; ++i) {
        double p = c[i]*d[i];
        ra2[i] = rb2[i-1] + p;
        rb2[i] = ra2[i] + c[i]*e[i];
    }
    /* 80-bit exact-ish prefix T */
    long double *Tarr = malloc(n*sizeof(long double));
    Tarr[0] = 0.0L;
    long double tsum = 0.0L;
    for (int64_t i = 1; i < n; ++i) {
        tsum += (long double)c[i]*d[i] + (long double)c[i]*e[i];
        Tarr[i] = tsum;
    }
    /* E_ref = ref1 - (b0 + T) */
    double me = 0, mer1 = 0, mer2 = 0; int64_t we = -1, wer1 = -1, wer2 = -1;
    for (int64_t i = 1; i < n; ++i) {
        double d = fabs(rb[i] - (double)((double)b[0] + Tarr[i]));
        double r = fmax(fabs(ra[i]), fabs(rb[i]));
        if (d > me) { me = d; we = i; }
        double q1 = d / (1.57e-7*r), q2 = d / 1.1145e-5;
        if (q1 > mer1) { mer1 = q1; wer1 = i; }
        if (q2 > mer2) { mer2 = q2; wer2 = i; }
    }
    printf("n=%ld T=%d\n", (long)n, T);
    printf("E_ref(ref1 vs exact): maxE=%.3e@%ld (|ref|=%.4g) ratio_rtol=%.2f@%ld (|ref|=%.4g) ratio_atol=%.2f@%ld (|ref|=%.4g)\n",
           me, (long)we, (we>0?fabs(rb[we]):-1), mer1, (long)wer1, (wer1>0?fabs(rb[wer1]):-1), mer2, (long)wer2, (wer2>0?fabs(rb[wer2]):-1));

    b0glob = b[0]; cg = c; dg = d; eg = e; NG = n;
    CS = malloc(8*(size_t)T*16);
    long double *CSld = malloc(sizeof(long double)*(size_t)T*16);
    ag = a; bg = b;

    run_v1(n);  report("v1  vs ref1(fma) ", ra, rb, n);
    run_v1(n);  report("v1  vs ref2(nofma)", ra2, rb2, n);
    run_v5(n);  report("v5  vs ref1(fma) ", ra, rb, n);
    run_v5(n);  report("v5  vs ref2(nofma)", ra2, rb2, n);
    run_v6(n, CSld); report("v6  vs ref1(fma) ", ra, rb, n);
    run_v6(n, CSld); report("v6  vs ref2(nofma)", ra2, rb2, n);
    /* leave correct output: ref1 (fma) */
    for (int64_t i = 0; i < n; ++i) { a[i] = ra[i]; b[i] = rb[i]; }
    fflush(stdout);
}
