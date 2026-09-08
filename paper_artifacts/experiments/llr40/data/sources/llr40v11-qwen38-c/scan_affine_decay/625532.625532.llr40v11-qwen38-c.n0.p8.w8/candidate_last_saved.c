#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define G 32768            /* micro-block size */
#define MAXT 24

static double *sA = NULL;    /* [MAXT][G] prefix products (up to j0 per micro-block) */
static double *sZ = NULL;    /* [MAXT][G] zero-carry values */
static double *blkA = NULL;  /* per micro-block A_k */
static double *blkB = NULL;  /* per micro-block B_k */
static double *carv = NULL;  /* per micro-block carry-in */
static long long *j0blk = NULL; /* where A_k prefix underflowed to 0 */
static size_t  snk = 0;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
}

static void ensure(size_t nk) {
    if (nk <= snk) return;
    free(blkA); free(blkB); free(carv); free(j0blk);
    if (!sA) {
        sA = (double *)calloc(sizeof(double) * MAXT * G, 1);
        sZ = (double *)calloc(sizeof(double) * MAXT * G, 1);
    }
    snk = nk;
    blkA = (double *)malloc(sizeof(double) * nk);
    blkB = (double *)malloc(sizeof(double) * nk);
    carv = (double *)malloc(sizeof(double) * nk);
    j0blk = (long long *)malloc(sizeof(long long) * nk);
    memset(blkA, 0, sizeof(double) * nk);
    memset(blkB, 0, sizeof(double) * nk);
    memset(carv, 0, sizeof(double) * nk);
    memset(j0blk, 0, sizeof(long long) * nk);
}

static void run_scan(const double *restrict c, const double *restrict x,
                     double *restrict y, long long n, long long K,
                     double *tp1, double *tp2) {
    double carry = 0.0;
    *tp1 = 0.0; *tp2 = 0.0;
    #pragma omp parallel
    {
        int t = omp_get_thread_num();
        int T = omp_get_num_threads();
        double *abA = &sA[(size_t)t * G];
        double *abZ = &sZ[(size_t)t * G];
        long long nw = (K + T - 1) / T;

        for (long long w = 0; w < nw; w++) {
            long long kw0 = w * T, kw1 = kw0 + T;
            if (kw1 > K) kw1 = K;

            /* Pass 1: (A,Z) chains; A chain dropped after exact underflow to 0. */
            double w0 = now_ms();
            #pragma omp for schedule(static)
            for (long long k = kw0; k < kw1; k++) {
                long long b0 = k * G;
                long long b1 = b0 + G;
                if (b1 > n) b1 = n;
                double a = 1.0, z = 0.0;
                long long j = 0;
                for (long long i = b0; i < b1 && a != 0.0; i++, j++) {
                    double ci = c[i];
                    a = ci * a;
                    z = ci * z + x[i];
                    abA[j] = a;
                    abZ[j] = z;
                }
                for (long long i = b0 + j; i < b1; i++, j++) {
                    z = c[i] * z + x[i];
                    abZ[j] = z;
                }
                blkA[k] = a;
                blkB[k] = z;
                j0blk[k] = j - (b1 - b0);  /* = first j where a==0; 0 if none */
            }
            double w1 = now_ms();

            #pragma omp single
            {
                double v = carry;
                for (long long k = kw0; k < kw1; k++) {
                    carv[k] = v;
                    v = blkA[k] * v + blkB[k];
                }
                carry = v;
            }

            /* Pass 2: y = A_i * v + Z_i where a survived; y = Z_i after. */
            double w2 = now_ms();
            #pragma omp for schedule(static)
            for (long long k = kw0; k < kw1; k++) {
                long long b0 = k * G;
                long long b1 = b0 + G;
                if (b1 > n) b1 = n;
                long long j0 = j0blk[k];
                double v = carv[k];
                long long i = 0;
                for (; i < j0; i++)
                    y[b0 + i] = abA[i] * v + abZ[i];
                for (; i < (b1 - b0); i++)
                    y[b0 + i] = abZ[i];
            }
            double w3 = now_ms();
            if (t == 0) { *tp1 += w1 - w0; *tp2 += w3 - w2; }
        }
    }
}

void scan_affine_decay_fp64(const double *restrict c, const double *restrict x,
                            double *restrict y, const int64_t LEN_1D) {
    long long n = (long long)LEN_1D;
    if (n <= 0) return;
    if (n == 1) { y[0] = x[0]; return; }
    if (n < (1LL << 20)) {
        y[0] = x[0];
        for (long long i = 1; i < n; i++) y[i] = c[i] * y[i - 1] + x[i];
        return;
    }

    long long K = (n + G - 1) / G;
    ensure((size_t)K);

    static int ncall = 0;
    ncall++;
    int reps = (ncall == 1) ? 3 : 1;
    for (int rr = 0; rr < reps; rr++) {
        double t0 = now_ms(), p1, p2;
        run_scan(c, x, y, n, K, &p1, &p2);
        double t1 = now_ms();
        if (rr == reps - 1 || ncall == 1)
            printf("call=%d rep=%d p1=%.3f p2=%.3f other=%.3f total=%.3f ms\n",
                   ncall, rr, p1, p2, (t1 - t0) - p1 - p2, t1 - t0);
    }
    fflush(stdout);
}
