/* C implementation of the 3D Laplacian stencil kernel.
   The reference Python kernel (laplacian_stencil_3d_numpy.py) computes
   a high‑order 8th‑order finite‑difference Laplacian on a periodic grid
   and the per‑state kinetic energy.
   This C version follows the same algorithm and uses the same calling
   convention as the other generated reference kernels in the repository.
   The function signature matches the pattern used by the other kernels:
       void laplacian_stencil_3d_fp64(const double *restrict psi,
                                      double *restrict lap,
                                      double *restrict ekin,
                                      const int64_t N,
                                      const int64_t k,
                                      const double inv_h2);
   Arguments:
     psi   – input wavefunctions, shape (N,N,N,k), flattened in C order.
     lap   – output Laplacian, same shape as psi.
     ekin  – per‑state kinetic energy, length k.
     N     – grid size along each spatial dimension.
     k     – number of wavefunctions (states).
     inv_h2 – 1/h^2, where h = 0.2 (grid spacing).
*/

#define _USE_MATH_DEFINES
#include <stddef.h>
#include <stdint.h>
#include <math.h>

void laplacian_stencil_3d_fp64(const double *restrict psi,
                                            double *restrict lap,
                                            double *restrict ekin,
                                            const int64_t N,
                                            const int64_t k,
                                            const double inv_h2)
{
    /* 8th‑order central‑difference coefficients for the second derivative. */
    const double C0 = -205.0 / 72.0;                     /* central coefficient */
    const double CW[4] = {8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0};

    /* Initialise kinetic‑energy accumulator. */
    for (int64_t s = 0; s < k; ++s) {
        ekin[s] = 0.0;
    }

    /* Compute Laplacian and kinetic energy. */
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        int64_t ip[4];
        int64_t im[4];
        for (int m = 1; m <= 4; ++m) {
            ip[m-1] = i + m;
            if (ip[m-1] >= N) ip[m-1] -= N;
            im[m-1] = i - m;
            if (im[m-1] < 0) im[m-1] += N;
        }
        for (int64_t j = 0; j < N; ++j) {
            int64_t jp[4];
            int64_t jm[4];
            for (int m = 1; m <= 4; ++m) {
                jp[m-1] = j + m;
                if (jp[m-1] >= N) jp[m-1] -= N;
                jm[m-1] = j - m;
                if (jm[m-1] < 0) jm[m-1] += N;
            }
            for (int64_t l = 0; l < N; ++l) {
                int64_t lp[4];
                int64_t lm[4];
                for (int m = 1; m <= 4; ++m) {
                    lp[m-1] = l + m;
                    if (lp[m-1] >= N) lp[m-1] -= N;
                    lm[m-1] = l - m;
                    if (lm[m-1] < 0) lm[m-1] += N;
                }

                size_t base = ((size_t)i * (size_t)N + (size_t)j) * (size_t)N + (size_t)l;
                size_t base_off = base * (size_t)k;

                /* Pre‑compute neighbor base offsets (without the per‑state stride). */
                size_t nb0p[4], nb0m[4];
                size_t nb1p[4], nb1m[4];
                size_t nb2p[4], nb2m[4];
                for (int m = 0; m < 4; ++m) {
                    nb0p[m] = ((size_t)ip[m] * (size_t)N + (size_t)j) * (size_t)N + (size_t)l;
                    nb0m[m] = ((size_t)im[m] * (size_t)N + (size_t)j) * (size_t)N + (size_t)l;
                    nb1p[m] = ((size_t)i * (size_t)N + (size_t)jp[m]) * (size_t)N + (size_t)l;
                    nb1m[m] = ((size_t)i * (size_t)N + (size_t)jm[m]) * (size_t)N + (size_t)l;
                    nb2p[m] = ((size_t)i * (size_t)N + (size_t)j) * (size_t)N + (size_t)lp[m];
                    nb2m[m] = ((size_t)i * (size_t)N + (size_t)j) * (size_t)N + (size_t)lm[m];
                }

                /* Compute all states at this grid point. */
                #pragma omp simd
                for (int64_t s = 0; s < k; ++s) {
                    double acc = 3.0 * C0 * psi[base_off + s];
                    for (int m = 0; m < 4; ++m) {
                        acc += CW[m] * (psi[nb0p[m] * (size_t)k + s] + psi[nb0m[m] * (size_t)k + s]);
                        acc += CW[m] * (psi[nb1p[m] * (size_t)k + s] + psi[nb1m[m] * (size_t)k + s]);
                        acc += CW[m] * (psi[nb2p[m] * (size_t)k + s] + psi[nb2m[m] * (size_t)k + s]);
                    }
                    double lap_val = inv_h2 * acc;
                    lap[base_off + s] = lap_val;
                    ekin[s] += psi[base_off + s] * lap_val;
                }
            }
        }
    }

    /* Finalise kinetic energy. */
    for (int64_t s = 0; s < k; ++s) {
        ekin[s] *= -0.5;
    }
}
