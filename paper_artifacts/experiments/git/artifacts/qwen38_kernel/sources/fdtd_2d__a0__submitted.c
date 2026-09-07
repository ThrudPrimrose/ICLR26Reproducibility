#include <stdint.h>

/* FDTD 2D update, 3-statement per timestep:
   ey[0,:]=fict[t]; ey[1:,:]-=eyc*(hz[1:,:]-hz[:-1,:]);
   ex[:,1:]-=exc*(hz[:,1:]-hz[:,:-1]);
   hz[:-1,:-1]-=hzc*((ex[:-1,1:]-ex[:-1,:-1])+(ey[1:,:-1])-ey[:-1,:-1])
   Statement 1&2 write disjoint arrays, both read OLD hz -> independent.
   Statement 3 depends on both. Same per-element expression as reference. */
void fdtd_2d_fp64(double *restrict ex, double *restrict ey,
                  const double *restrict fict, double *restrict hz,
                  int64_t NX, int64_t NY, int64_t TMAX,
                  double ex_courant, double ey_courant, double hz_courant) {
  for (int64_t t = 0; t < TMAX; ++t) {
    const double f = fict[t];
    #pragma omp parallel
    {
      #pragma omp for schedule(static)
      for (int64_t j = 0; j < NY; ++j)
        ey[j] = f;

      #pragma omp for schedule(static)
      for (int64_t i = 1; i < NX; ++i) {
        double *eyr = ey + i * NY;
        const double *hzr = hz + i * NY;
        const double *hzm = hz + (i - 1) * NY;
        for (int64_t j = 0; j < NY; ++j)
          eyr[j] -= ey_courant * (hzr[j] - hzm[j]);
      }

      #pragma omp for schedule(static)
      for (int64_t i = 0; i < NX; ++i) {
        double *exr = ex + i * NY;
        const double *hzr = hz + i * NY;
        for (int64_t j = 1; j < NY; ++j)
          exr[j] -= ex_courant * (hzr[j] - hzr[j - 1]);
      }

      #pragma omp for schedule(static)
      for (int64_t i = 0; i < NX - 1; ++i) {
        double *hzr = hz + i * NY;
        const double *exr = ex + i * NY;
        const double *eyr = ey + i * NY;
        const double *eyrn = ey + (i + 1) * NY;
        for (int64_t j = 0; j < NY - 1; ++j)
          hzr[j] -= hz_courant * (((exr[j + 1] - exr[j]) + eyrn[j]) - eyr[j]);
      }
    }
  }
}
