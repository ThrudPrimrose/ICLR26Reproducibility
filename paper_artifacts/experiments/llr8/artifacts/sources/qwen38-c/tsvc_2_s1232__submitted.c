/* TSVC s1232: aa[i,j] = bb[i,j] + cc[i,j] for i >= j*VLEN, row-major (i*N+j).
 *
 * Optimizations vs the strided reference loop:
 *  1. Loop interchange: iterate rows i (contiguous in memory), inner loop over
 *     j = 0..i/VLEN+1 is contiguous -> vectorizes (AVX-512).
 *  2. OpenMP over rows with sqrt-partitioned ranges: per-row work w(i)=i/V+1
 *     grows with i, so equal-RANGE split would strand 4 of 5 threads; split on
 *     equal CUMULATIVE work C(n) = n^2/(2V) + n*(2V-1)/(2V) instead.
 */
#include <stdint.h>
#include <math.h>
#include <omp.h>

/* Inverse of C(n) = n^2/(2V) + n*(2V-1)/(2V): solve n^2 + n(2V-1) - 2VS = 0. */
static int64_t row_bound(double V, double S)
{
    const double a = 2.0 * V - 1.0;
    double n = (-a + sqrt(a * a + 8.0 * V * S)) * 0.5;
    if (n < 0.0) n = 0.0;
    return (int64_t)(n + 0.5);
}

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb,
                       const double *restrict cc, int64_t LEN_2D, int64_t VLEN)
{
    const int64_t N = LEN_2D;
    if (N <= 0) return;

    int P = (int)omp_get_max_threads();
    if (P < 1) P = 1;
    if ((int64_t)P > N) P = (int)N;

#pragma omp parallel num_threads(P)
    {
        const int64_t t = omp_get_thread_num();
        const int64_t Pt = omp_get_num_threads();

        int64_t lo, hi;
        if (VLEN > 0) {
            /* per-row work w(i) = i/VLEN + 1; cumulative C(n) as above */
            const double V = (double)VLEN;
            const double total = (double)N * (double)N / (2.0 * V)
                               + (double)N * (2.0 * V - 1.0) / (2.0 * V);
            lo = row_bound(V, total * (double)t / (double)Pt);
            hi = row_bound(V, total * (double)(t + 1) / (double)Pt);
            if (lo < 0) lo = 0;
            if (lo > N) lo = N;
            if (hi < 0) hi = 0;
            if (hi > N) hi = N;
        } else {
            /* VLEN <= 0: i >= j*VLEN for every j -> the whole array */
            lo = (N * t) / Pt;
            hi = (N * (t + 1)) / Pt;
        }

        for (int64_t i = lo; i < hi; ++i) {
            int64_t jmax = (VLEN > 0) ? (i / VLEN + 1) : N;
            double *a = aa + i * N;
            const double *b = bb + i * N;
            const double *c = cc + i * N;
            for (int64_t j = 0; j < jmax; ++j)
                a[j] = b[j] + c[j];
        }
    }
}
