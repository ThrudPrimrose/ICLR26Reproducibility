#include <stdint.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    const int64_t n = LEN_1D;
    double tA = omp_get_wtime();
    if (n <= 0) { printf("n=0\n"); fflush(stdout); return; }
    int T = omp_get_max_threads();
    if (T < 4) T = 4;
    if (T > n) T = (int)n;
    printf("n=%lld T=%d OMP_NUM_THREADS=%s\n", (long long)n, T, getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "(unset)");
    fflush(stdout);
    double t0 = omp_get_wtime();
    if (n < 200000 || T < 2) {
        double sum = 0.0;
        for (int64_t i = 0; i < n; ++i) { sum += a[i]; b[i] = sum; }
    } else {
        double *tot = (double *)malloc((size_t)T * sizeof(double));
#pragma omp parallel num_threads(T)
        {
#pragma omp for schedule(static)
            for (int64_t t = 0; t < (int64_t)T; ++t) {
                int64_t base = n / T, rem = n % T;
                int64_t i0 = t * base + (t < rem ? t : rem);
                int64_t i1 = i0 + base + (t < rem ? 1 : 0);
                double s = 0.0;
                for (int64_t i = i0; i < i1; ++i) { s += a[i]; b[i] = s; }
                tot[t] = s;
            }
        }
        double t1 = omp_get_wtime();
        double acc = 0.0;
        for (int64_t t = 0; t < (int64_t)T; ++t) { double e = acc; acc += tot[t]; tot[t] = e; }
#pragma omp parallel num_threads(T)
        {
#pragma omp for schedule(static)
            for (int64_t t = 0; t < (int64_t)T; ++t) {
                double e = tot[t];
                if (e != 0.0) {
                    int64_t base = n / T, rem = n % T;
                    int64_t i0 = t * base + (t < rem ? t : rem);
                    int64_t i1 = i0 + base + (t < rem ? 1 : 0);
                    for (int64_t i = i0; i < i1; ++i) b[i] += e;
                }
            }
        }
        double t2 = omp_get_wtime();
        free(tot);
        printf("phase1=%.1f ms phase2+3=%.1f ms\n", (t1-t0)*1e3, (t2-t1)*1e3);
        fflush(stdout);
    }
    double t3 = omp_get_wtime();
    double d = 0.0;
#pragma omp target map(tofrom: d)
    d = d + 1.0;
    double t4 = omp_get_wtime();
    (void)d;
    printf("target=%.1f us total=%.1f ms tA2t0=%.1f us\n", (t4-t3)*1e6, (t4-tA)*1e3, (t0-tA)*1e6);
    fflush(stdout);
}
