/* Optimized 2-D FDTD update (Yee grid).
 *
 * Per timestep the four updates are:
 *   ey[0,:]      = fict[t]
 *   ey[1:,:]    -= ey_courant * (hz[1:, :] - hz[:-1, :])
 *   ex[:, 1:]   -= ex_courant * (hz[:, 1:] - hz[:, :-1])
 *   hz[:-1,:-1] -= hz_courant * (ex[:-1,1:] - ex[:-1,:-1] + ey[1:,:-1] - ey[:-1,:-1])
 *
 * Phase A (first three) only reads hz and writes ex/ey, so it is one
 * independent elementwise phase; phase B updates hz from the fresh ex/ey.
 * The elementwise math and its rounding order are exactly the reference's.
 *
 * Threading: one persistent OpenMP team for the whole time loop.  Each
 * thread owns a fixed contiguous block of grid rows for every timestep
 * (strided row distribution causes heavy cross-socket L2/L3 traffic here,
 * so the blocks are assigned once and reused).  Phase A needs the previous
 * timestep's hz everywhere, phase B needs phase A everywhere, hence two
 * barriers per timestep.  Expression order matches the reference exactly,
 * so the results are bit-identical.
 */
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey, const double *restrict fict, double *restrict hz, const int64_t NX, const int64_t NY, const int64_t TMAX, const double ex_courant, const double ey_courant, const double hz_courant) {
    if (TMAX <= 0 || NX <= 0 || NY <= 0) return;

    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const int64_t i0 = (NX * tid) / nt;
        const int64_t i1 = (NX * (tid + 1)) / nt;

        for (int64_t t = 0; t < TMAX; ++t) {
            const double f = fict[t];

            /* Phase A: ex and ey only read hz. */
            if (i0 == 0) {
                double *ey_row = ey;
                for (int64_t si1 = 0; si1 < NY; ++si1) ey_row[si1] = f;
            }
            for (int64_t si0 = i0; si0 < i1; ++si0) {
                double *ey_row = ey + si0 * NY;
                if (si0 != 0) {
                    const double *hz_row = hz + si0 * NY;
                    const double *hz_rowm1 = hz + (si0 - 1) * NY;
                    for (int64_t si1 = 0; si1 < NY; ++si1)
                        ey_row[si1] -= (ey_courant * (hz_row[si1] - hz_rowm1[si1]));
                }
                const double *hz_row = hz + si0 * NY;
                double *ex_row = ex + si0 * NY;
                for (int64_t si1 = 1; si1 < NY; ++si1)
                    ex_row[si1] -= (ex_courant * (hz_row[si1] - hz_row[si1 - 1]));
            }

            /* Barrier: phase B must see every row's fresh ex/ey
             * (it reads the first row of the next block as a halo). */
            #pragma omp barrier

            /* Phase B: hz interior uses the updated ex/ey. */
            const int64_t iB = i1 < (NX - 1) ? i1 : (NX - 1);
            for (int64_t si0 = i0; si0 < iB; ++si0) {
                const double *ex_row = ex + si0 * NY;
                const double *ey_row = ey + si0 * NY;
                const double *ey_rowp1 = ey + (si0 + 1) * NY;
                double *hz_row = hz + si0 * NY;
                for (int64_t si1 = 0; si1 < (NY - 1); ++si1)
                    hz_row[si1] -= (hz_courant * (((ex_row[si1 + 1] - ex_row[si1]) + ey_rowp1[si1]) - ey_row[si1]));
            }

            /* Barrier: next timestep's phase A must see every block's new hz. */
            #pragma omp barrier
        }
    }
}
