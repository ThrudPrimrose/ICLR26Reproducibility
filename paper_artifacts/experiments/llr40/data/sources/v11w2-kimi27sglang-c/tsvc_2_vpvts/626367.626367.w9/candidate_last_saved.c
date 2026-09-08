#include <stdint.h>
#include <cblas.h>

void tsvc_2_vpvts_fp64(double *restrict a, const double *restrict b,
                       const int64_t LEN_1D, const int64_t S) {
    const int64_t n = LEN_1D;
    if (n <= 0) return;

    if (n < 2048) {
        const double s = (double)S;
        for (int64_t i = 0; i < n; ++i) a[i] += b[i] * s;
        return;
    }

    static int threads_set = 0;
    if (!threads_set) {
        openblas_set_num_threads(24);
        threads_set = 1;
    }

    cblas_daxpy((int32_t)n, (double)S, b, 1, a, 1);
}
