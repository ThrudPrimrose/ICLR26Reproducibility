/* FDTD 2D update. Matches fdtd_2d_reference.c / numpy reference semantics:
   per time step: ey[0,:]=fict[t]; ey[1:,:]-=cey*(hz[1:,:]-hz[:-1,:]);
   ex[:,1:]-=cex*(hz[:,1:]-hz[:,:-1]);
   hz[:-1,:-1]-=chz*(ex[:-1,1:]-ex[:-1,:-1]+ey[1:,:-1]-ey[:-1,:-1]).
   The ey and ex phases both read only the old hz and write disjoint arrays,
   so they are fused into one parallel pass; the hz pass must see the new
   ex/ey, hence the barrier (plain `omp for`).
*/
#include <stdint.h>
#include <stddef.h>

void fdtd_2d_fp64(double *restrict ex, double *restrict ey,
                  const double *restrict fict, double *restrict hz,
                  int64_t NX, int64_t NY, int64_t TMAX,
                  double ex_courant, double ey_courant, double hz_courant)
{
    const int nxi = (int)NX, nyi = (int)NY, tmax = (int)TMAX;

#pragma omp parallel
    {
        for (int t = 0; t < tmax; ++t) {
            const double fv = fict[t];
#pragma omp for schedule(static)
            for (int i = 0; i < nxi; ++i) {
                double *exr = ex + (size_t)i * nyi;
                const double *hr = hz + (size_t)i * nyi;
                if (i == 0) {
                    double *ey0 = ey;
                    for (int j = 0; j < nyi; ++j) ey0[j] = fv;
                } else {
                    double *eyr = ey + (size_t)i * nyi;
                    const double *hp = hz + (size_t)(i - 1) * nyi;
                    for (int j = 0; j < nyi; ++j)
                        eyr[j] -= ey_courant * (hr[j] - hp[j]);
                }
                for (int j = 1; j < nyi; ++j)
                    exr[j] -= ex_courant * (hr[j] - hr[j - 1]);
            }
#pragma omp for schedule(static)
            for (int i = 0; i < nxi - 1; ++i) {
                double *hr = hz + (size_t)i * nyi;
                const double *exr = ex + (size_t)i * nyi;
                const double *ey0 = ey + (size_t)i * nyi;
                const double *ey1 = ey + (size_t)(i + 1) * nyi;
                for (int j = 0; j < nyi - 1; ++j)
                    hr[j] -= hz_courant * ((exr[j + 1] - exr[j]) + (ey1[j] - ey0[j]));
            }
        }
    }
}
