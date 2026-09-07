/* FDTD 2D update -- OpenMP parallelized over grid rows.
 * Per time step: phases A(ey)+B(ex) only read old hz (fused into one
 * parallel loop), phase C(hz) needs the fresh ex/ey (separate barrier).
 */
#include <stdint.h>
#include <stddef.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey,
                  const double *restrict fict, double *restrict hz,
                  int64_t NX, int64_t NY, int64_t TMAX,
                  double ex_courant, double ey_courant, double hz_courant)
{
    #pragma omp parallel
    for (int64_t t = 0; t < TMAX; ++t) {
        /* A: ey update (row 0 is the boundary) + B: ex update.
         * Both consume the old hz only, so they are independent and
         * can run in the same work-sharing region. */
        #pragma omp for
        for (int64_t i = 0; i < NX; ++i) {
            double *ei = ey + (size_t)i * NY;
            const double *hi = hz + (size_t)i * NY;
            if (i == 0) {
                for (int64_t j = 0; j < NY; ++j) ei[j] = fict[t];
            } else {
                const double *hm = hz + ((size_t)i - 1) * NY;
                for (int64_t j = 0; j < NY; ++j)
                    ei[j] -= ey_courant * (hi[j] - hm[j]);
            }
            double *xi = ex + (size_t)i * NY;
            for (int64_t j = 1; j < NY; ++j)
                xi[j] -= ex_courant * (hi[j] - hi[j - 1]);
        }
        /* C: hz update using the fresh ex/ey. */
        #pragma omp for
        for (int64_t i = 0; i < NX - 1; ++i) {
            double *hi = hz + (size_t)i * NY;
            const double *xi = ex + (size_t)i * NY;
            const double *ei = ey + (size_t)i * NY;
            const double *en = ey + (size_t)(i + 1) * NY;
            for (int64_t j = 0; j < NY - 1; ++j)
                hi[j] -= hz_courant * (((xi[j + 1] - xi[j]) + en[j]) - ei[j]);
        }
    }
}
