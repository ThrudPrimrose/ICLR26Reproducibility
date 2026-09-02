#include <stdint.h>
#include <omp.h>

void gesummv_fp64(const double *restrict A, const double *restrict B,
                  double *restrict out, const double *restrict x,
                  int64_t N, double alpha, double beta) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; i++) {
        const double *arow = A + i * N;
        const double *brow = B + i * N;
        double s1 = 0.0, s2 = 0.0;
        for (int64_t j = 0; j < N; j++) {
            s1 += arow[j] * x[j];
            s2 += brow[j] * x[j];
        }
        out[i] = alpha * s1 + beta * s2;
    }
}
