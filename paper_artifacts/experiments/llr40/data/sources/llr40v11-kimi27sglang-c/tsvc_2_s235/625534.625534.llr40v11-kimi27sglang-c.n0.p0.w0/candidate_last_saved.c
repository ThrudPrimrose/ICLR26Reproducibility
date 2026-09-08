#include <stdint.h>
#include <omp.h>

void tsvc_2_s235_fp64(double *restrict a, double *restrict aa,
                      const double *restrict b, const double *restrict bb,
                      const double *restrict c, const int64_t LEN_2D) {
    const int64_t N = LEN_2D;
    const int64_t BS = 512;

    #pragma omp parallel for schedule(static)
    for (int64_t ib = 0; ib < N; ib += BS) {
        int64_t iend = ib + BS;
        if (iend > N) iend = N;

        for (int64_t i = ib; i < iend; ++i) {
            a[i] += b[i] * c[i];
        }

        for (int64_t j = 1; j < N; ++j) {
            const double *restrict bbr = bb + j * N;
            double *restrict aar = aa + j * N;
            const double *restrict aar_prev = aa + (j - 1) * N;
            #pragma omp simd
            for (int64_t i = ib; i < iend; ++i) {
                aar[i] = aar_prev[i] + bbr[i] * a[i];
            }
        }
    }
}
