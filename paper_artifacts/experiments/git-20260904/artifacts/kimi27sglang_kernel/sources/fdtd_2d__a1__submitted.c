#include <stdint.h>
#include <stdlib.h>
#include <math.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict, double *restrict hz, int64_t NX, int64_t NY, int64_t TMAX, double ex_courant, double ey_courant, double hz_courant) {
    #pragma omp parallel
    for (int64_t t = 0; t < TMAX; ++t) {
        #pragma omp for schedule(static)
        for (int64_t si1 = 0; si1 < NY; ++si1) {
            ey[(0)*(NY) + (si1)] = fict[t];
        }

        #pragma omp for schedule(static)
        for (int64_t si0 = 1; si0 < NX; ++si0) {
            for (int64_t si1 = 0; si1 < NY; ++si1) {
                ey[(si0)*(NY) + (si1)] -= ey_courant * (hz[(si0)*(NY) + (si1)] - hz[((si0 - 1))*(NY) + (si1)]);
            }
        }

        #pragma omp for schedule(static)
        for (int64_t si0 = 0; si0 < NX; ++si0) {
            for (int64_t si1 = 1; si1 < NY; ++si1) {
                ex[(si0)*(NY) + (si1)] -= ex_courant * (hz[(si0)*(NY) + (si1)] - hz[(si0)*(NY) + ((si1 - 1))]);
            }
        }

        #pragma omp for schedule(static)
        for (int64_t si0 = 0; si0 < (NX - 1); ++si0) {
            for (int64_t si1 = 0; si1 < (NY - 1); ++si1) {
                hz[(si0)*(NY) + (si1)] -= hz_courant * ((ex[(si0)*(NY) + ((si1 + 1))] - ex[(si0)*(NY) + (si1)]) + ey[((si0 + 1))*(NY) + (si1)] - ey[(si0)*(NY) + (si1)]);
            }
        }
    }
}
