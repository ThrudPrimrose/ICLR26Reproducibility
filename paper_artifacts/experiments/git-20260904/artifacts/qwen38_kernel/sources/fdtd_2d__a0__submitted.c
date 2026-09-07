/* FDTD-2D (PolyBenchC-4.2.1 adapted) -- optimized for HPCAgent-Bench.
 *
 * Per time step the reference does three disjoint array passes:
 *   1. ey[0,:]=fict[t]; ey[1:,:] -= c_ey*(hz[1:,:]-hz[:-1,:])
 *   2. ex[:,1:]       -= c_ex*(hz[:,1:]-hz[:,:-1])
 *   3. hz[:-1,:-1]    -= c_hz*((ex[:-1,1:]-ex[:-1,:-1])+(ey[1:,:-1]-ey[:-1,:-1]))
 * Passes 1 and 2 both only READ hz and write different arrays, so their row
 * updates are fused into one parallel loop (one row of ex + one row of ey per
 * iteration).  Pass 3 needs the freshly updated ex and ey, so it follows the
 * implicit barrier of the first omp for.
 *
 * Each per-row update lives in a small noinline helper that receives the row
 * pointers as separate restrict arguments: the rows are genuinely disjoint,
 * so the restrict assumption is sound, and the GCC vectorizer (which cannot
 * prove disjointness for pointers derived from the same array inside one
 * function) happily emits AVX-512 for the inner j loops.
 */
#include <stdint.h>

static void __attribute__((noinline))
row_ex(double *restrict exr, const double *restrict hzr, int64_t n, double c)
{
    for (int64_t j = 1; j < n; ++j)
        exr[j] -= c * (hzr[j] - hzr[j - 1]);
}

static void __attribute__((noinline))
row_ey(double *restrict eyr, const double *restrict hzr, const double *restrict hzp,
       int64_t n, double c)
{
    for (int64_t j = 0; j < n; ++j)
        eyr[j] -= c * (hzr[j] - hzp[j]);
}

static void __attribute__((noinline))
row_hz(double *restrict hzr, const double *restrict exr, const double *restrict eyr,
       const double *restrict eyp, int64_t n, double c)
{
    for (int64_t j = 0; j < n; ++j)
        hzr[j] -= c * (((exr[j + 1] - exr[j]) + eyp[j]) - eyr[j]);
}

void fdtd_2d_fp64(double *restrict ex, double *restrict ey,
                  const double *restrict fict, double *restrict hz,
                  int64_t NX, int64_t NY, int64_t TMAX,
                  double ex_courant, double ey_courant, double hz_courant)
{
    const int64_t NXm1 = NX - 1;
    const int64_t NYm1 = NY - 1;

#pragma omp parallel
    {
        for (int64_t t = 0; t < TMAX; ++t) {
            double *restrict e0 = ey;
            const double f = fict[t];
            for (int64_t j = 0; j < NY; ++j)
                e0[j] = f;

            /* Stage A+B: row i of ex (cols 1..) and row i of ey (all cols) */
#pragma omp for schedule(static)
            for (int64_t i = 0; i < NX; ++i) {
                row_ex(ex + i * NY, hz + i * NY, NY, ex_courant);
                if (i > 0)
                    row_ey(ey + i * NY, hz + i * NY, hz + (i - 1) * NY, NY,
                           ey_courant);
            }

            /* Stage C: interior hz update (needs the updated ex and ey) */
#pragma omp for schedule(static)
            for (int64_t i = 0; i < NXm1; ++i)
                row_hz(hz + i * NY, ex + i * NY, ey + i * NY,
                       ey + (i + 1) * NY, NYm1, hz_courant);
        }
    }
}
