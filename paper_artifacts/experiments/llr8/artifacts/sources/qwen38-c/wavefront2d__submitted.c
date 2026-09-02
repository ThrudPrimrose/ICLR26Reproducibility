#include <stdint.h>
#include <omp.h>
#include <math.h>

/* Blocked wavefront with FMA-contricted inner loop.
 * x_j = 0.25*x_{j-1} + 0.25*(r[j]+p[j]+p[j-1]); chain = one FMA.
 */
void wavefront2d_fp64(double *a, const int64_t LEN_2D,
                      uint8_t *restrict workspace, const int64_t workspace_size)
{
    (void)workspace; (void)workspace_size;
    const int64_t L = LEN_2D;
    if (L < 2) return;
    const int64_t B = 128;
    const int64_t NW = L - 1;
    const int64_t NB = (NW + B - 1) / B;
    #pragma omp parallel
    {
      for (int64_t d = 0; d <= 2 * (NB - 1); d++) {
        int64_t bi0 = d - (NB - 1); if (bi0 < 0) bi0 = 0;
        int64_t bi1 = d; if (bi1 > NB - 1) bi1 = NB - 1;
        #pragma omp for
        for (int64_t bi = bi0; bi <= bi1; bi++) {
          int64_t bj = d - bi;
          int64_t i0 = 1 + bi * B; if (i0 > L-1) i0 = L-1;
          int64_t i1 = i0 + B - 1; if (i1 > L-1) i1 = L-1;
          int64_t j0 = 1 + bj * B; if (j0 > L-1) j0 = L-1;
          int64_t j1 = j0 + B - 1; if (j1 > L-1) j1 = L-1;
          for (int64_t i = i0; i <= i1; i++) {
            double *r = a + i * L;
            const double *p = r - L;
            double x = r[j0-1];
            #pragma GCC unroll 4
            for (int64_t j = j0; j <= j1; j++) {
              double t = 0.25 * (r[j] + p[j] + p[j-1]);
              x = fma(0.25, x, t);
              r[j] = x;
            }
          }
        }
      }
    }
}
