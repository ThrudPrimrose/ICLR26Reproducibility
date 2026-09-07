#include <stdint.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey,
                  const double *restrict fict, double *restrict hz,
                  int64_t NX, int64_t NY, int64_t TMAX,
                  double ex_courant, double ey_courant, double hz_courant)
{
    const int64_t nx = NX;
    const int64_t ny = NY;

    #pragma omp parallel
    {
        for (int64_t t = 0; t < TMAX; ++t) {
            /* ey[0, :] = fict[t] */
            #pragma omp for
            for (int64_t j = 0; j < ny; ++j)
                ey[j] = fict[t];

            /* ey[i, :] -= ey_courant * (hz[i, :] - hz[i-1, :]), i >= 1 */
            #pragma omp for
            for (int64_t i = 1; i < nx; ++i) {
                double *restrict ey_i = ey + i * ny;
                const double *restrict hz_i = hz + i * ny;
                const double *restrict hz_im1 = hz + (i - 1) * ny;
                for (int64_t j = 0; j < ny; ++j)
                    ey_i[j] -= ey_courant * (hz_i[j] - hz_im1[j]);
            }

            /* ex[i, :] -= ex_courant * (hz[i, :] - hz[i, :-1]) */
            #pragma omp for
            for (int64_t i = 0; i < nx; ++i) {
                double *restrict ex_i = ex + i * ny;
                const double *restrict hz_i = hz + i * ny;
                for (int64_t j = 1; j < ny; ++j)
                    ex_i[j] -= ex_courant * (hz_i[j] - hz_i[j - 1]);
            }

            /* hz[i, :] -= hz_courant * (ex[i,1:] - ex[i,:-1]
                                        + ey[i+1, :] - ey[i, :]) */
            #pragma omp for
            for (int64_t i = 0; i < nx - 1; ++i) {
                double *restrict hz_i = hz + i * ny;
                const double *restrict ex_i = ex + i * ny;
                const double *restrict ey_i = ey + i * ny;
                const double *restrict ey_ip1 = ey + (i + 1) * ny;
                for (int64_t j = 0; j < ny - 1; ++j)
                    hz_i[j] -= hz_courant * ((ex_i[j + 1] - ex_i[j])
                                            + (ey_ip1[j] - ey_i[j]));
            }
        }
    }
}
