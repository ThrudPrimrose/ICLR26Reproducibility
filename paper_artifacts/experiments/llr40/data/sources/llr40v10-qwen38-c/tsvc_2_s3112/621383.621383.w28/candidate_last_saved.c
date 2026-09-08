#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

void tsvc_2_s3112_fp64(const double *restrict a, double *restrict b, const int64_t LEN_1D) {
    const int T = 8;
    int64_t chunk = (LEN_1D + T - 1) / T;
    double *S = (double*)malloc(T * sizeof(double));
    #pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        int64_t st = (int64_t)tid * chunk, en = st + chunk; if (en > LEN_1D) en = LEN_1D;
        double r = 0.0;
        for (int64_t i = st; i < en; ++i) r += a[i];
        S[tid] = r;
    }
    double o = 0.0;
    for (int t = 0; t < T; ++t) { double c = o; o += S[t]; S[t] = c; }
    #pragma omp parallel for num_threads(T)
    for (int t = 0; t < T; ++t) {
        int64_t st = (int64_t)t * chunk, en = st + chunk; if (en > LEN_1D) en = LEN_1D;
        double run = S[t];
        for (int64_t i = st; i < en; ++i) { run += a[i]; b[i] = run; }
    }
    free(S);
}
