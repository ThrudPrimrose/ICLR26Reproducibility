#include <stdint.h>
#include <omp.h>

void tsvc_2_s1232_fp64(double *restrict aa, const double *restrict bb, const double *restrict cc,
                       const int64_t LEN_2D, const int64_t VLEN) {
    if (LEN_2D <= 0 || VLEN <= 0) {
        return;
    }

    const size_t N = (size_t)LEN_2D;
    const size_t V = (size_t)VLEN;

    #pragma omp parallel for schedule(guided,1)
    for (size_t i = 0; i < N; ++i) {
        const size_t jlim = (V == 1) ? i : (i / V);
        const size_t jmax = (jlim < N) ? jlim : N - 1;

        double *restrict a_row = aa + i * N;
        const double *restrict b_row = bb + i * N;
        const double *restrict c_row = cc + i * N;

        #pragma omp simd simdlen(8)
        for (size_t j = 0; j <= jmax; ++j) {
            a_row[j] = b_row[j] + c_row[j];
        }
    }
}
