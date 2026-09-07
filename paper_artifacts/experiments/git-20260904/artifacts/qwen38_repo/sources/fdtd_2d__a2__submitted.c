/* Optimized 2-D FDTD time-domain update.
 *
 * Structure per timestep:
 *   Phase A (independent of each other, hz read-only):
 *     ey[0, :]   = fict[t]
 *     ey[i, :]  -= ey_courant * (hz[i, j] - hz[i-1, j])      for i >= 1
 *     ex[i, j+1] -= ex_courant * (hz[i, j+1] - hz[i, j])     for all i
 *   Phase B (needs the phase-A ex/ey values):
 *     hz[i, j]  -= hz_courant * (((ex[i, j+1] - ex[i, j]) + ey[i+1, j]) - ey[i, j])
 *
 * A persistent OpenMP team runs the whole TMAX loop; inside each timestep the
 * row loops are work-shared (OMP for + one barrier between phases).  The inner
 * j loops are plain contiguous-row updates that -O3 -march=native vectorizes
 * with AVX-512.  Per-element operation order matches the reference exactly.
 */
#include <stdint.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict, double *restrict hz, const int64_t NX, const int64_t NY, const int64_t TMAX, const double ex_courant, const double ey_courant, const double hz_courant) {
    if (TMAX <= 0 || NX <= 0 || NY <= 0) return;
    const int64_t nxm1 = NX - 1;
    const int64_t nym1 = NY - 1;

    #pragma omp parallel
    {
        for (int64_t t = 0; t < TMAX; ++t) {
            const double f = fict[t];
            /* Phase A: ey and ex updates (hz is read-only here). */
            #pragma omp for nowait
            for (int64_t i = 0; i < NX; ++i) {
                double *eyp = ey + i * NY;
                if (i == 0) {
                    for (int64_t j = 0; j < NY; ++j) eyp[j] = f;
                } else {
                    const double *hzr  = hz + i * NY;
                    const double *hzr1 = hz + (i - 1) * NY;
                    for (int64_t j = 0; j < NY; ++j)
                        eyp[j] -= ey_courant * (hzr[j] - hzr1[j]);
                }
                double *exp = ex + i * NY;
                const double *hzi = hz + i * NY;
                for (int64_t j = 0; j < nym1; ++j)
                    exp[j + 1] -= ex_courant * (hzi[j + 1] - hzi[j]);
            }
            /* All ey/ex updates must be visible before any hz update reads them. */
            #pragma omp barrier
            /* Phase B: hz update (interior only).
             * No `nowait`: the implicit end-of-loop barrier also orders the
             * next timestep's phase A after this hz update. */
            #pragma omp for
            for (int64_t i = 0; i < nxm1; ++i) {
                double *hzp  = hz + i * NY;
                const double *exp = ex + i * NY;
                const double *eyp = ey + i * NY;
                const double *eyp1 = ey + (i + 1) * NY;
                for (int64_t j = 0; j < nym1; ++j)
                    hzp[j] -= hz_courant * (((exp[j + 1] - exp[j]) + eyp1[j]) - eyp[j]);
            }
        }
    }
}
