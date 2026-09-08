#include <stdint.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b, const double *restrict c,
                      const double *restrict d, const double *restrict e,
                      const int64_t LEN_1D) {
    double sum = 0.0;
    #pragma omp parallel for simd reduction(+:sum) aligned(a,b,c,d,e:64) schedule(static)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        double ci = c[i];
        double ai = ci + d[i];
        double bi = ci + e[i];
        a[i] = ai;
        b[i] = bi;
        sum += ai + bi;
    }
    b[0] = sum;
}
