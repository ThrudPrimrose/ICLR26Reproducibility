#include <stdint.h>
#include <math.h>
#include <omp.h>

#define MAXT 512

void scan_affine_decay_fp64(const double *restrict c, const double *restrict x,
                            double *restrict y, const int64_t LEN_1D)
{
    if (LEN_1D <= 0) return;
    y[0] = x[0];
    if (LEN_1D == 1) return;

    /* Elements 1..LEN_1D-1 form a scan over the affine monoid:
     *   step i:  v -> c[i]*v + x[i]
     * composed:  (A,B) op (a,b) = (A*a, A*b+B)
     * Chunk map F_k:  y[e_k-1] = A_k * y[s_k-1] + B_k
     * where A_k = prod_{i in chunk} c[i], B_k = chunk scan starting from seed 0.
     */
    int64_t n = LEN_1D - 1;   /* number of steps (elements 1..LEN_1D-1) */

    static double A_arr[MAXT], B_arr[MAXT];
    int nthreads = 0;
    int64_t start[MAXT];   /* chunk start element index (1-based start of steps) */

    /* ---- pass 1: per-chunk affine pairs ---- */
#pragma omp parallel
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
        nthreads = nt;
        int64_t base = n / nt, rem = n % nt;
        int64_t s = 1 + t * base + (t < rem ? t : rem);
        int64_t e = 1 + (t + 1) * base + (t + 1 < rem ? t + 1 : rem);
        if (e > s) {
            const double *cc = c + s;
            const double *xx = x + s;
            double A = 1.0, B = 0.0;
            int64_t i = 0, cnt = e - s;
            for (; i < cnt; i++) {
                B = fma(cc[i], B, xx[i]);
                A *= cc[i];
            }
            A_arr[t] = A;
            B_arr[t] = B;
        } else {
            A_arr[t] = 1.0;
            B_arr[t] = 0.0;
        }
    }

    /* ---- serial prefix over chunks (nthreads steps) ---- */
    double carry = x[0];
    for (int k = 0; k < nthreads; k++) {
        start[k] = start[k]; /* chunk k starts at start[k] */
        double A = A_arr[k], B = B_arr[k];
        /* recompute nothing; store the carry for chunk k in B slot? keep separate */
        A_arr[k] = carry;            /* repurpose A_arr[k] as carry-in for chunk k */
        carry = A * carry + B;
    }

    /* ---- pass 2: true recurrence from chunk carry ---- */
#pragma omp parallel
    {
        int t = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int64_t s = 1 + t * (n / nt) + (t < (n % nt) ? t : n % nt);
        int64_t e = 1 + (t + 1) * (n / nt) + (t + 1 < (n % nt) ? t + 1 : n % nt);
        double v = A_arr[t];   /* carry-in (reused slot) */
        const double *cc = c + s;
        const double *xx = x + s;
        double *yy = y + s;
        int64_t i = 0, cnt = e - s;
        for (; i < cnt; i++) {
            v = fma(cc[i], v, xx[i]);
            yy[i] = v;
        }
    }
}
