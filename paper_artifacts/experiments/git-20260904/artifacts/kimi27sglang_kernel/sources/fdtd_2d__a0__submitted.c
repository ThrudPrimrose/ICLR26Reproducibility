#include <stdint.h>
#include <omp.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey,
                  const double *restrict fict, double *restrict hz,
                  int64_t NX, int64_t NY, int64_t TMAX,
                  double ex_courant, double ey_courant, double hz_courant) {
    for (int64_t t = 0; t < TMAX; ++t) {
        const double ft = fict[t];
        #pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < NY; ++j) {
            ey[j] = ft;
        }

        #pragma omp parallel for schedule(static)
        for (int64_t i = 1; i < NX; ++i) {
            double *restrict ey_row = ey + i * NY;
            const double *restrict hz_row = hz + i * NY;
            const double *restrict hz_prev = hz + (i - 1) * NY;
            for (int64_t j = 0; j < NY; ++j) {
                ey_row[j] -= ey_courant * (hz_row[j] - hz_prev[j]);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < NX; ++i) {
            double *restrict ex_row = ex + i * NY;
            const double *restrict hz_row = hz + i * NY;
            for (int64_t j = 1; j < NY; ++j) {
                ex_row[j] -= ex_courant * (hz_row[j] - hz_row[j - 1]);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < NX - 1; ++i) {
            const double *restrict ex_row = ex + i * NY;
            const double *restrict ey_row = ey + i * NY;
            const double *restrict ey_next = ey + (i + 1) * NY;
            double *restrict hz_row = hz + i * NY;
            for (int64_t j = 0; j < NY - 1; ++j) {
                hz_row[j] -= hz_courant *
                             ((ex_row[j + 1] - ex_row[j]) +
                              (ey_next[j] - ey_row[j]));
            }
        }
    }
}
