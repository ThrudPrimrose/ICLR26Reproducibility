/* Hand-optimized 2-D FDTD kernel (replaces the autogen seed version).
 *
 * Per time step the three updates keep the reference's exact data flow:
 *   ey_new uses OLD hz,  ex_new uses OLD hz,  hz_new uses NEW ex/ey.
 * So per step: (1) update ey and ex from the untouched hz, (2) after an
 * implicit barrier update hz from the fresh ex/ey.  Both stages are
 * elementwise independent across grid rows and vectorize cleanly.
 */
#include <stdint.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey,
                  const double *restrict fict, double *restrict hz,
                  const int64_t NX, const int64_t NY, const int64_t TMAX,
                  const double ex_courant, const double ey_courant,
                  const double hz_courant) {
    for (int64_t t = 0; t < TMAX; ++t) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < NX; ++i) {
            double *const ey_row = ey + i * NY;
            double *const ex_row = ex + i * NY;
            const double *const hz_row = hz + i * NY;
            if (i == 0) {
                for (int64_t j = 0; j < NY; ++j) ey_row[j] = fict[t];
            } else {
                const double *const hz_up = hz_row - NY;
                for (int64_t j = 0; j < NY; ++j)
                    ey_row[j] -= ey_courant * (hz_row[j] - hz_up[j]);
            }
            for (int64_t j = 1; j < NY; ++j)
                ex_row[j] -= ex_courant * (hz_row[j] - hz_row[j - 1]);
        }
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < NX - 1; ++i) {
            double *const hz_row = hz + i * NY;
            const double *const ex_row = ex + i * NY;
            const double *const ey_row = ey + i * NY;
            const double *const ey_dn = ey_row + NY;
            for (int64_t j = 0; j + 1 < NY; ++j)
                hz_row[j] -= hz_courant * ((ex_row[j + 1] - ex_row[j]) + (ey_dn[j] - ey_row[j]));
        }
    }
}
