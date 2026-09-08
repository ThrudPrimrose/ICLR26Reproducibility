#include <stdint.h>
#include <omp.h>

void tsvc_2_s319_fp64(double *restrict a, double *restrict b,
                      const double *restrict c, const double *restrict d,
                      const double *restrict e, const int64_t LEN_1D) {
    double *restrict aa = __builtin_assume_aligned(a, 64);
    double *restrict bb = __builtin_assume_aligned(b, 64);
    const double *restrict cc = __builtin_assume_aligned(c, 64);
    const double *restrict dd = __builtin_assume_aligned(d, 64);
    const double *restrict ee = __builtin_assume_aligned(e, 64);

    double sum = 0.0;
    #pragma omp parallel for simd reduction(+:sum) schedule(static) aligned(aa,bb,cc,dd,ee:64)
    for (int64_t i = 0; i < LEN_1D; ++i) {
        const double ai = cc[i] + dd[i];
        const double bi = cc[i] + ee[i];
        aa[i] = ai;
        bb[i] = bi;
        sum += ai + bi;
    }
    b[0] = sum;
}
